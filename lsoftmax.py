import os
import math
import mxnet as mx
import numpy as np
import time

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'


class LSoftmaxOp(mx.operator.CustomOp):
    '''LSoftmax from <Large-Margin Softmax Loss for Convolutional Neural Networks>
    '''

    def __init__(self, margin, beta):
        self.margin = int(margin)
        self.beta = float(beta)
        self.c_map = []
        self.k_map = []
        c_m_n = lambda m, n: math.factorial(n) / math.factorial(m) / math.factorial(n-m)
        for i in range(margin+1):
            self.c_map.append(c_m_n(i, margin))
            self.k_map.append(math.cos(i * math.pi / margin))
        
        # save them in forward operation, used in backward 
        self.k = None
        self.cos_t = None
        self.cos_mt = None
        self.w_norm_choose = None
        self.x_norm = None
        self.w_choose = None
    
    def l2_norm_eachrow(self, input_data):
        '''
            compute norm of each row
            2-dim matrix only
        '''
        out = mx.nd.sqrt(mx.nd.sum(mx.nd.square(input_data), axis=1))
        return out

    # reshape the array from (n,) -> (n,1) , damn it..
    def expand_axis(self, arr):
        return arr.reshape((arr.shape[0], 1))
    
    # adjust some var for broadcast, from (n,) to (n,1)
    def adjust_var_shapes(self):
        self.w_norm_choose = self.expand_axis(self.w_norm_choose)
        self.x_norm = self.expand_axis(self.x_norm)
        self.k = self.expand_axis(self.k)
        self.cos_t = self.expand_axis(self.cos_t)
        self.cos_mt = self.expand_axis(self.cos_mt)
        
    def find_k(self, cos_t):
        '''find k for cos(theta)
        '''
        # for numeric issue
        eps = 1e-5
        le = lambda x, y: x < y or abs(x-y) < eps
        for i in range(self.margin):
            if le(self.k_map[i+1], cos_t) and le(cos_t, self.k_map[i]):
                return i
        raise ValueError('can not find k for cos_t = %f'%cos_t)

    def calc_cos_mt(self, cos_t):
        '''calculate cos(m*theta), u are so cute @luoyetx :)
        '''
        cos_mt = math.cos(self.margin*math.acos(np.clip(cos_t, -1.0, 1.0)))
        return cos_mt

    def forward(self, is_train, req, in_data, out_data, aux):
        assert len(in_data) == 3
        assert len(out_data) == 1
        assert len(req) == 1
        x, label, w = in_data

        out = mx.nd.dot(x, w.T)
        w_norm = self.l2_norm_eachrow(w)
        self.x_norm = self.l2_norm_eachrow(x)

        self.w_norm_choose = w_norm.broadcast_to(shape=( label.shape[0], w_norm.shape[0]))
        self.w_norm_choose = mx.nd.choose_element_0index(self.w_norm_choose, label)

        f = mx.ndarray.choose_element_0index(out, label)
        self.cos_t = f / (self.w_norm_choose * self.x_norm)
        cos_t = self.cos_t.asnumpy()
        
        k = np.zeros_like(cos_t)
        cos_mt = np.zeros_like(cos_t)
        # how to apply selfdefined function to ndarray?
        for i in range(label.shape[0]):
            k[i] = self.find_k(cos_t[i])
            cos_mt[i] = self.calc_cos_mt(cos_t[i])

        # go back to gpu, is it ok with multiple gpu?
        self.k = mx.nd.array(k, ctx=x.context)
        self.cos_mt = mx.nd.array(cos_mt, ctx=x.context)

        f_new = (mx.ndarray.power(-1, self.k) * self.cos_mt - 2*self.k) * (self.w_norm_choose*self.x_norm)
        out = mx.nd.fill_element_0index(out, (self.beta*f_new+f)/(1+self.beta), label)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert len(in_data) == 3
        assert len(out_grad) == 1
        assert len(in_grad) == 3
        assert len(req) == 3
        x, label, w = in_data
        n = label.shape[0]
        margin = self.margin
        o_grad = out_grad[0]
        x_grad = mx.nd.dot(o_grad, w)
        w_grad = mx.nd.dot(o_grad.T, x)

        power = mx.nd.power # used a lot ...

        # adjust shape for broadcast
        self.adjust_var_shapes()
        sin2_t = 1 - mx.nd.square(self.cos_t)

        # allocate momory for the first time
        if self.w_choose is None:
            self.w_choose = x.copy()
            self.w_choose[:] = 0
        
        # equivalence of mshadow's 'take' function here ? 
        w_numpy = w.asnumpy()
        self.w_choose = mx.nd.array(w_numpy[label.astype(np.int32).asnumpy(),:], ctx=x.context)
        
        # gradient wrt to x
        dcos_dx = self.w_choose/(self.w_norm_choose*self.x_norm) - \
                x*self.cos_t/(mx.nd.square(self.x_norm))
        dsin2_dx = -2*self.cos_t*dcos_dx
        dcosm_dx = margin*power(self.cos_t, margin-1)*dcos_dx # p == 0
        
        for p in range(1, margin/2 + 1):
            dcosm_dx += pow(-1,p)*self.c_map[2*p]*( (margin-2*p)*power(self.cos_t, margin-2*p-1)*power(sin2_t, p)*dcos_dx + \
                        p*power(sin2_t, p-1)*power(self.cos_t, margin-2*p)*dsin2_dx)
        df_dx = (power(-1, self.k)*self.cos_mt - 2*self.k)*self.w_norm_choose/self.x_norm*x + \
                self.w_norm_choose*self.x_norm*( power(-1,self.k)*dcosm_dx)
        alpha = self.beta / (1 + self.beta)

        grad_scale = mx.nd.choose_element_0index(o_grad, label)
        x_grad += alpha*self.expand_axis(grad_scale)*(df_dx-self.w_choose)
        
        # gradient wrt to w
        dcos_dw = x/(self.x_norm*self.w_norm_choose) - \
                    self.w_choose*self.cos_t/(mx.nd.square(self.w_norm_choose))
        dsin2_dw = -2*self.cos_t*dcos_dw
        dcosm_dw = margin*power(self.cos_t, margin-1)*dcos_dw # p == 0
        for p in range(1, margin/2 + 1):
            dcosm_dw += pow(-1,p)*self.c_map[2*p]*((margin-2*p)*power(self.cos_t,margin-2*p-1)*power(sin2_t,p)*dcos_dw + \
                       p*power(self.cos_t, margin-2*p)*power(sin2_t, p-1)*dsin2_dw)
        
        df_dw = (power(-1,self.k)*self.cos_mt-2*self.k)*self.x_norm/self.w_norm_choose*self.w_choose + \
                power(-1,self.k)*self.x_norm*self.w_norm_choose*dcosm_dw

        alpha = self.beta / (1 + self.beta)
        grad_scale = mx.nd.choose_element_0index(o_grad,label)
        df_dw = alpha*self.expand_axis(grad_scale)*(df_dw-x)
        
        # no take function ...damn, use numpy for the job
        w_grad_numpy = w_grad.asnumpy()
        w_grad_numpy[label.astype(np.int32).asnumpy()] += df_dw.asnumpy()
        w_grad[:] = mx.nd.array(w_grad_numpy, ctx=x.context)

        self.assign(in_grad[0], req[0], x_grad)
        self.assign(in_grad[2], req[2], w_grad)


@mx.operator.register("LSoftmax")
class LSoftmaxProp(mx.operator.CustomOpProp):

    def __init__(self, num_hidden, beta, margin):
        super(LSoftmaxProp, self).__init__(need_top_grad=True)
        self.margin = int(margin)
        self.num_hidden = int(num_hidden)
        self.beta = float(beta)

    def list_arguments(self):
        return ['data', 'label', 'weight']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert len(in_shape) == 3, "LSoftmaxOp input data: [data, label, weight]"
        dshape = in_shape[0]
        lshape = in_shape[1]
        assert len(dshape) == 2, "data shape should be (batch_size, feature_dim)"
        assert len(lshape) == 1, "label shape should be (batch_size,)"
        wshape = (self.num_hidden, dshape[1])
        oshape = (dshape[0], self.num_hidden)
        return [dshape, lshape, wshape], [oshape,], []

    def create_operator(self, ctx, shapes, dtypes):
        return LSoftmaxOp(margin=self.margin, beta=self.beta)


def test_op():
    """test LSoftmax Operator
    """
    # build symbol
    batch_size = cmd_args.batch_size
    embedding_dim = cmd_args.embedding_dim
    num_classes = cmd_args.num_classes
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    weight = mx.sym.Variable('weight')
    args = {
        'data': np.random.normal(0, 1, (batch_size, embedding_dim)),
        'weight': np.random.normal(0, 1, (num_classes, embedding_dim)),
        'label': np.random.choice(num_classes, batch_size),
    }

    if cmd_args.op_impl == 'py':
        symbol = mx.sym.Custom(data=data, label=label, weight=weight, num_hidden=10,
                               beta=cmd_args.beta, margin=cmd_args.margin,
                               op_type='LSoftmax', name='lsoftmax')
    else:
        symbol = mx.sym.LSoftmax(data=data, label=label, weight=weight, num_hidden=num_classes,
                                 margin=cmd_args.margin, beta=cmd_args.beta, name='lsoftmax')

    data_shape = (batch_size, embedding_dim)
    label_shape = (batch_size,)
    weight_shape = (num_classes, embedding_dim)
    ctx = mx.cpu() if cmd_args.op_impl == 'py' else mx.gpu()
    executor = symbol.simple_bind(ctx=ctx, data=data_shape, label=label_shape, weight=weight_shape)

    def forward(data, label, weight):
        data = mx.nd.array(data, ctx=ctx)
        label = mx.nd.array(label, ctx=ctx)
        weight = mx.nd.array(weight, ctx=ctx)
        executor.forward(is_train=True, data=data, label=label, weight=weight)
        return executor.output_dict['lsoftmax_output'].asnumpy()

    def backward(out_grad):
        executor.backward(out_grads=[mx.nd.array(out_grad, ctx=ctx)])
        return executor.grad_dict

    def gradient_check(name, i, j):
        '''gradient check on x[i, j]
        '''
        eps = 1e-4
        threshold = 1e-2
        reldiff = lambda a, b: abs(a-b) / (abs(a) + abs(b))
        # calculate by backward
        output = forward(data=args['data'], weight=args['weight'], label=args['label'])
        grad_dict = backward(output)
        grad = grad_dict[name].asnumpy()[i, j]
        # calculate by \delta f / 2 * eps
        loss = lambda x: np.square(x).sum() / 2
        args[name][i, j] -= eps
        loss1 = loss(forward(data=args['data'], weight=args['weight'], label=args['label']))
        args[name][i, j] += 2 * eps
        loss2 = loss(forward(data=args['data'], weight=args['weight'], label=args['label']))
        grad_expect = (loss2 - loss1) / (2 * eps)
        # check
        rel_err = reldiff(grad_expect, grad)
        if rel_err > threshold:
            print 'gradient check failed'
            print 'expected %lf given %lf, relative error %lf'%(grad_expect, grad, rel_err)
            return False
        else:
            print 'gradient check pass'
            return True

    # test forward
    output = forward(data=args['data'], weight=args['weight'], label=args['label'])
    diff = args['data'].dot(args['weight'].T) - output

    # test backward
    # gradient check on data
    data_gc_pass = 0
    for i in range(args['data'].shape[0]):
        for j in range(args['data'].shape[1]):
            print 'gradient check on data[%d, %d]'%(i, j)
            if gradient_check('data', i, j):
                data_gc_pass += 1
    # gradient check on weight
    weight_gc_pass = 0
    for i in range(args['weight'].shape[0]):
        for j in range(args['weight'].shape[1]):
            print 'gradient check on weight[%d, %d]'%(i, j)
            if gradient_check('weight', i, j):
                weight_gc_pass += 1
    print '===== Summary ====='
    print 'gradient on data pass ratio is %lf'%(float(data_gc_pass) / args['data'].size)
    print 'gradient on weight pass ratio is %lf'%(float(weight_gc_pass) / args['weight'].size)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="test batch size")
    parser.add_argument('--num-classes', type=int, default=10, help="test number of classes")
    parser.add_argument('--embedding-dim', type=int, default=3, help="test embedding dimension")
    parser.add_argument('--margin', type=int, default=2, help="test lsoftmax margin")
    parser.add_argument('--beta', type=float, default=10, help="test lsoftmax beta")
    parser.add_argument('--op-impl', type=str, choices=['py', 'cpp'], default='py', help="test op implementation")
    cmd_args = parser.parse_args()
    print cmd_args

    # check
    if cmd_args.op_impl == 'cpp':
        try:
            op_creator = mx.sym.LSoftmax
        except AttributeError:
            print 'No cpp operator for LSoftmax, Skip test'
            import sys
            sys.exit(0)

    test_op()
