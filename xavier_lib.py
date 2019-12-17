import torch
import torch.nn
import numpy as np


# Varies for different model
# mapping the name of conv to its corresponding bn
def mapping(conv_name):
    try:
        return conv_name[:-1] + str(int(conv_name[-1])+1)
    except ValueError:
        return 'None'


class InfoStruct(object):

    def __init__(self, module, pre_f_cls, f_cls, b_cls):

        # init
        self.module = module
        self.pre_f_cls = pre_f_cls
        self.f_cls = f_cls
        self.b_cls = b_cls

        # forward statistic
        self.forward_mean = None
        self.variance = None
        self.forward_cov = None
        self.channel_num = None
        self.zero_variance_masked_zero = None
        self.zero_variance_masked_one = None
        self.alpha = None  # variance after de-correlation
        self.sorted_alpha_index = None
        self.stack_op_for_weight = None

        # backward statistic
        self.grad_mean = None
        self.grad_cov = None
        self.adjust_matrix = None

        # score
        self.pure_score = None
        self.score = None
        self.sorted_index = None

        # parameters form model
        self.weight = None
        self.adjusted_weight = None

        # assigned before pruning
        self.bn_module = None

    def compute_statistic_and_fetch_weight(self):
        # compute forward covariance
        self.forward_mean = self.f_cls.sum_mean / self.f_cls.counter
        self.forward_cov = (self.f_cls.sum_covariance / self.f_cls.counter) - \
            torch.mm(self.forward_mean.view(-1, 1), self.forward_mean.view(1, -1))

        self.channel_num = list(self.forward_cov.shape)[0]

        # equal 0 where variance of an activate is 0
        self.variance = torch.diag(self.forward_cov)
        self.zero_variance_masked_zero = torch.sign(self.variance)

        # where 0 var compensate 1
        self.zero_variance_masked_one = torch.nn.Parameter(torch.ones(self.channel_num, dtype=torch.double),
                                                           requires_grad=False).cuda() - \
            self.zero_variance_masked_zero
        repaired_forward_cov = self.forward_cov + torch.diag(self.zero_variance_masked_one)

        f_cov_inverse = repaired_forward_cov.inverse().to(torch.float)
        repaired_alpha = torch.reciprocal(torch.diag(f_cov_inverse))
        self.alpha = repaired_alpha *  self.zero_variance_masked_zero.to(torch.float)

        self.stack_op_for_weight = (f_cov_inverse.t() * repaired_alpha.view(1, -1)).t()

        self.weight = self.module.weight.detach()

        self.grad_mean = self.b_cls.sum_mean / self.b_cls.counter
        self.grad_cov = (self.b_cls.sum_covariance / self.b_cls.counter) - \
            torch.mm(self.grad_mean.view(-1, 1), self.grad_mean.view(1, -1))
        eig_value, eig_vec = torch.eig(self.grad_cov, eigenvectors=True)

        self.adjust_matrix = torch.mm(torch.diag(torch.sqrt(eig_value[:, 0])), eig_vec.t()).to(torch.float)
        compare = torch.mm(torch.diag(torch.sqrt(eig_value[:, 0])), eig_vec.t())
        print(self.f_cls.name)
        print('var', torch.sign(torch.diag(self.forward_cov)))
        vars = torch.abs(torch.sign(torch.diag(self.grad_cov)))
        summm = torch.abs(torch.sign(self.grad_mean))
        print('mask', vars+summm)

        self.pure_score = torch.sum(torch.pow(torch.squeeze(self.weight), 2), dim=0) * self.alpha
        self.sorted_alpha_index = torch.argsort(self.pure_score)

        self.adjusted_weight = torch.mm(self.adjust_matrix, torch.squeeze(self.weight))

        self.score = torch.sum(torch.pow(self.adjusted_weight, 2), dim=0) * self.alpha
        self.sorted_index = torch.argsort(self.score)
        # print(torch.sort(self.score)[0])

    def clear_zero_variance(self):

        # according to zero variance mask, remove all the channels with 0 variance,
        # this function first update [masks] in pre_forward_hook,
        # then update parameters in [bn module] or biases in the last layer

        verify = int(torch.sum(self.pre_f_cls.read_data() - self.zero_variance_masked_zero.to(torch.float) * self.pre_f_cls.read_data()))

        if verify != 0:

            print('Number of more zero var channel: ', verify)

            self.pre_f_cls.load_data(self.zero_variance_masked_zero.to(torch.float))

            # print('remove activate: ', torch.sum(self.zero_variance_masked_one))

            used_mean = self.forward_mean.to(torch.float) * self.zero_variance_masked_one.to(torch.float)
            repair_base = torch.squeeze(torch.mm(torch.squeeze(self.weight), used_mean.view(-1, 1)))

            if self.bn_module is None:
                print('Modify biases in', self.module)
                self.module.bias.data -= repair_base
            else:
                self.bn_module.running_mean.data -= repair_base

    def minimum_score(self) -> [int, float]:

        channel_mask = self.pre_f_cls.read_data()
        for index in list(np.array(self.sorted_index.cpu())):
            index = int(index)
            if int(channel_mask[index]) != 0:
                return index, float(self.score[index])

    def minimum_var_score(self) -> [int, float, float]:
        # return the score of the minimum de-correlated variance
        # by this way, we can achieve better generality
        channel_mask = self.pre_f_cls.read_data()

        for index in list(np.array(self.sorted_alpha_index.cpu())):
            index = int(index)
            if int(channel_mask[index]) != 0:
                return index, float(self.score[index]), float(self.pure_score[index])

    def prune_then_modify(self, index_of_channel):
        # update [mask]
        channel_mask = self.pre_f_cls.read_data()
        channel_mask[index_of_channel] = 0
        self.pre_f_cls.load_data(channel_mask)

        # update [bn]
        if self.f_cls.dim == 4:
            connections = self.weight[:, index_of_channel, 0, 0]
        else:
            connections = self.weight[:, index_of_channel]
        repair_base = connections * self.forward_mean[index_of_channel]

        if self.bn_module is None:
            print('Modify biases in', self.module)
            self.module.bias.data -= repair_base
        else:
            self.bn_module.running_mean.data -= repair_base
            repair_var = connections * self.variance[index_of_channel]
            self.bn_module.running_var.data -= repair_var

        # update [weights]
        before_update = np.array(self.weight.cpu().detach())

        new_aw = self.adjusted_weight - \
                 torch.mm(self.adjusted_weight[:, index_of_channel].view(-1, 1),
                          self.stack_op_for_weight[index_of_channel, :].view(1, -1))

        new_weight = torch.mm(torch.inverse(self.adjust_matrix), new_aw)

        if self.f_cls.dim == 4:
            self.weight[:, :, 0, 0] = new_weight
        else:
            self.weight[:, :] = new_weight
        self.module.weight.data = self.weight

        # update [scores]
        self.forward_cov[:, index_of_channel] = 0
        self.forward_cov[index_of_channel, :] = 0

        channel_masked_one = torch.nn.Parameter(torch.ones(self.channel_num, dtype=torch.float),
                                                requires_grad=False).cuda() - channel_mask
        repaired_forward_cov = self.forward_cov + torch.diag(channel_masked_one).to(torch.double)

        f_cov_inverse = repaired_forward_cov.inverse().to(torch.float)
        repaired_alpha = torch.reciprocal(torch.diag(f_cov_inverse))
        self.alpha = repaired_alpha * channel_mask

        self.stack_op_for_weight = (f_cov_inverse.t() * repaired_alpha.view(1, -1)).t()

        self.pure_score = torch.sum(torch.pow(torch.squeeze(self.weight), 2), dim=0) * self.alpha
        self.sorted_alpha_index = torch.argsort(self.pure_score)

        adjusted_weight = torch.mm(self.adjust_matrix, torch.squeeze(self.weight))

        self.score = torch.sum(torch.pow(adjusted_weight, 2), dim=0) * self.alpha
        self.sorted_index = torch.argsort(self.score)
        # print(torch.sort(self.score)[0])

        return repaired_forward_cov, before_update, self.adjust_matrix

    def local_prune_then_modify(self, index_of_channel):
        # update [mask]
        channel_mask = self.pre_f_cls.read_data()
        channel_mask[index_of_channel] = 0
        self.pre_f_cls.load_data(channel_mask)

        # update [bn]
        if self.f_cls.dim == 4:
            connections = self.weight[:, index_of_channel, 0, 0]
        else:
            connections = self.weight[:, index_of_channel]
        repair_base = connections * self.forward_mean[index_of_channel]

        if self.bn_module is None:
            print('Modify biases in', self.module)
            self.module.bias.data -= repair_base
        else:
            self.bn_module.running_mean.data -= repair_base
            repair_var = connections * self.variance[index_of_channel]
            self.bn_module.running_var.data -= repair_var

        # update [weights]
        before_update = np.array(self.weight.cpu().detach())

        new_weight = torch.squeeze(self.weight) - torch.mm(self.weight[:, index_of_channel].view(-1, 1),
                                                           self.stack_op_for_weight[index_of_channel, :].view(1, -1))
        if self.f_cls.dim == 4:
            self.weight[:, :, 0, 0] = new_weight
        else:
            self.weight[:, :] = new_weight
        self.module.weight.data = self.weight

        # update [scores]
        self.forward_cov[:, index_of_channel] = 0
        self.forward_cov[index_of_channel, :] = 0

        channel_masked_one = torch.nn.Parameter(torch.ones(self.channel_num, dtype=torch.float)).cuda() - \
                             channel_mask
        repaired_forward_cov = self.forward_cov + torch.diag(channel_masked_one).to(torch.double)

        f_cov_inverse = repaired_forward_cov.inverse().to(torch.float)
        repaired_alpha = torch.reciprocal(torch.diag(f_cov_inverse))
        self.alpha = repaired_alpha * channel_mask

        self.stack_op_for_weight = (f_cov_inverse.t() * repaired_alpha.view(1, -1)).t()

        self.pure_score = torch.sum(torch.pow(torch.squeeze(self.weight), 2), dim=0) * self.alpha
        self.sorted_alpha_index = torch.argsort(self.pure_score)

        adjusted_weight = torch.mm(self.adjust_matrix, torch.squeeze(self.weight))

        self.score = torch.sum(torch.pow(adjusted_weight, 2), dim=0) * self.alpha
        self.sorted_index = torch.argsort(self.score)
        # print(torch.sort(self.score)[0])

        return repaired_forward_cov, before_update, self.adjust_matrix

    def reset(self):
        self.f_cls.reset()
        self.b_cls.reset()

        # forward statistic
        self.forward_mean = None
        self.variance = None
        self.forward_cov = None
        self.channel_num = None
        self.zero_variance_masked_zero = None
        self.zero_variance_masked_one = None
        self.alpha = None  # variance after de-correlation
        self.stack_op_for_weight = None

        # backward statistic
        self.grad_mean = None
        self.grad_cov = None
        self.adjust_matrix = None

        # score
        self.score = None
        self.sorted_index = None

        # parameters form model
        self.weight = None

    def query_channel_num(self):

        channel_mask = self.pre_f_cls.read_data()

        return int(torch.sum(channel_mask).cpu()), int(channel_mask.shape[0])


def compute_statistic_and_update(samples, sum_mean, sum_covar, counter) -> None:
    samples = samples.to(torch.half).to(torch.double)
    samples_num = list(samples.shape)[0]
    counter += samples_num
    sum_mean += torch.sum(samples, dim=0)
    sum_covar += torch.mm(samples.permute(1, 0), samples)


class ForwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.channel_num = None
        self.sum_mean = None
        self.sum_covariance = None
        self.counter = None

    def __call__(self, module, inputs, output) -> None:
        with torch.no_grad():
            channel_num = list(inputs[0].shape)[1]
            self.channel_num = channel_num
            if self.sum_mean is None or self.sum_covariance is None:
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num).to(torch.double), requires_grad=False).cuda()
                self.sum_covariance = \
                    torch.nn.Parameter(torch.zeros(channel_num, channel_num).to(torch.double),
                                       requires_grad=False).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1).to(torch.double), requires_grad=False).cuda()
            # from [N,C,W,H] to [N*W*H,C]
            if self.dim == 4:
                samples = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = inputs[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)

    def reset(self):
        self.sum_mean = None
        self.sum_covariance = None
        self.counter = None


class BackwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.channel_num = None
        self.sum_covariance = None
        self.sum_mean = None
        self.counter = None

    def __call__(self, module, grad_input, grad_output) -> None:
        with torch.no_grad():
            channel_num = list(grad_output[0].shape)[1]
            self.channel_num = channel_num
            if self.sum_covariance is None:
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num).to(torch.double),
                                                   requires_grad=False).cuda()
                self.sum_covariance = \
                    torch.nn.Parameter(torch.zeros(channel_num, channel_num).to(torch.double),
                                       requires_grad=False).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1).to(torch.double), requires_grad=False).cuda()
            if self.dim == 4:
                samples = grad_output[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = grad_output[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)

    def reset(self):
        self.sum_mean = None
        self.sum_covariance = None
        self.counter = None


class PreForwardHook(object):

    def __init__(self, name, module, dim=4):
        self.name = name
        self.dim = dim
        self.module = module
        if dim == 4:
            self.channel_num = module.in_channels
        elif dim == 2:
            self.channel_num = module.in_features
        module.register_parameter('mask', torch.nn.Parameter(torch.ones(self.channel_num), requires_grad=False))

    def __call__(self, module, inputs):
        if self.dim == 4:
            modified = torch.mul(inputs[0].permute([0, 2, 3, 1]), module.mask)
            return modified.permute([0, 3, 1, 2])
        elif self.dim == 2:
            return torch.mul(inputs[0], module.mask)
        else:
            raise Exception

    def read_data(self):
        return self.module.mask.data

    def load_data(self, data):
        self.module.mask.data = data


class StatisticManager(object):

    def __init__(self):

        self.name_to_statistic = {}
        self.bn_name = {}

        self.tmp = [0, 0]

    def __call__(self, model):

        for name, sub_module in model.named_modules():

            if isinstance(sub_module, torch.nn.Conv2d):
                if sub_module.kernel_size[0] == 1:
                    pre_hook_cls = PreForwardHook(name, sub_module)
                    hook_cls = ForwardStatisticHook(name)
                    back_hook_cls = BackwardStatisticHook(name)
                    sub_module.register_forward_pre_hook(pre_hook_cls)
                    sub_module.register_forward_hook(hook_cls)
                    sub_module.register_backward_hook(back_hook_cls)
                    self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)
                # print('conv', name)

            elif isinstance(sub_module, torch.nn.Linear):
                pre_hook_cls = PreForwardHook(name, sub_module, dim=2)
                hook_cls = ForwardStatisticHook(name, dim=2)
                back_hook_cls = BackwardStatisticHook(name, dim=2)
                sub_module.register_forward_pre_hook(pre_hook_cls)
                sub_module.register_forward_hook(hook_cls)
                sub_module.register_backward_hook(back_hook_cls)
                self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)
                # print('conv', name)

            elif isinstance(sub_module, torch.nn.BatchNorm1d) or isinstance(sub_module, torch.nn.BatchNorm2d):
                self.bn_name[name] = sub_module
                # print('bn', name)

    def computer_score(self):

        with torch.no_grad():

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                if mapping(name) in self.bn_name:
                    info.bn_module = self.bn_name[mapping(name)]

                info.compute_statistic_and_fetch_weight()

                info.clear_zero_variance()

    def prune(self, pruned_num):

        for _ in range(pruned_num):

            min_score = 1000
            the_info = None
            the_name = None
            index = None

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                idx, score = info.minimum_score()
                if score < min_score:
                    min_score = score
                    the_info = info
                    the_name = name
                    index = idx

            print('pruned score: ', min_score, 'name: ', the_name)
            cov, weight, adj = the_info.prune_then_modify(index)
            if self.tmp[0] is not None:
                if self.tmp[0] == the_name:
                    import scipy.io as sio
                    sio.savemat('show.mat', {'before': np.array(self.tmp[1].cpu().detach()),
                                             'after': np.array(cov.cpu().detach()),
                                             'weight': weight,
                                             'adj_matrix': np.array(adj.cpu().detach())})
                    print('save')
            self.tmp[0] = the_name
            self.tmp[1] = cov

    def prune_local(self, pruned_num):

        for _ in range(pruned_num):

            min_score = 1000
            the_info = None
            the_name = None
            shooow = None
            index = None

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                idx, score, show = info.minimum_var_score()
                if score < min_score:
                    min_score = score
                    the_info = info
                    the_name = name
                    index = idx
                    shooow = show

            print('pruned score: ', min_score, 'name: ', the_name, 'pure score: ', shooow, 'index: ', index)
            cov, weight, adj = the_info.local_prune_then_modify(index)
            if self.tmp[0] is not None:
                if self.tmp[0] == the_name:
                    import scipy.io as sio
                    sio.savemat('show.mat', {'before': np.array(self.tmp[1].cpu().detach()),
                                             'after': np.array(cov.cpu().detach()),
                                             'weight': weight,
                                             'adj_matrix': np.array(adj.cpu().detach())})
                    print('save')
            self.tmp[0] = the_name
            self.tmp[1] = cov

    def pruning_overview(self):

        all_channel_num = 0
        remained_channel_num = 0

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]
            r, a = info.query_channel_num()
            all_channel_num += a
            remained_channel_num += r

        print('channel number: ', remained_channel_num, '/', all_channel_num)

    def reset(self):

        for name in self.name_to_statistic:

            info = self.name_to_statistic[name]

            info.reset()

    def visualize(self):

        from matplotlib import pyplot as plt
        i = 1
        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            forward_mean = info.f_cls.sum_mean / info.f_cls.counter
            forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))

            grad_mean = info.b_cls.sum_mean / info.b_cls.counter
            grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
            plt.subplot(10, 15, i)
            plt.imshow(np.array(forward_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            plt.subplot(10, 15, i)
            plt.imshow(np.array(grad_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            if i > 150:
                break
        plt.show()


class MaskManager(object):

    def __init__(self):
        pass

    def __call__(self, model):

        for name, sub_module in model.named_modules():

            if isinstance(sub_module, torch.nn.Conv2d):
                if sub_module.kernel_size[0] == 1:
                    pre_hook_cls = PreForwardHook(name, sub_module)
                    sub_module.register_forward_pre_hook(pre_hook_cls)

            elif isinstance(sub_module, torch.nn.Linear):
                pre_hook_cls = PreForwardHook(name, sub_module, dim=2)
                sub_module.register_forward_pre_hook(pre_hook_cls)
