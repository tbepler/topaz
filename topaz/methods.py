from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def autoencoder_loss(model, X):
    X = X.unsqueeze(1)
    z = model.features(X)
    score = model.classifier(z).view(-1)

    X_ = model.generative(z)
    pad = (model.width - model.generative.width)//2
    if pad > 0:
        X = X[:,:,pad:-pad,pad:-pad]
    recon_loss = (X - X_)**2
    recon_loss = torch.mean(torch.sum(recon_loss.view(X.size(0),-1), 1))

    return recon_loss, score

class PN:
    def __init__(self, model, optim, criteria, pi=None, l2=0
                , autoencoder=0):
        self.model = model
        self.optim = optim
        self.criteria = criteria
        self.pi = pi
        self.l2 = l2
        self.autoencoder = autoencoder

        self.header = ['loss', 'precision', 'adjusted_precision', 'tpr', 'fpr']
        if self.autoencoder > 0:
            self.header = ['loss', 'recon_error', 'precision', 'adjusted_precision', 'tpr', 'fpr']

    def step(self, X, Y):

        if self.autoencoder > 0:
            recon_error, score = autoencoder_loss(self.model, X)
        else:
            score = self.model(X).view(-1)

        if self.pi is not None:
            loss_one = self.criteria(score[Y==1], Y[Y==1])
            loss_zero = self.criteria(score[Y==0], Y[Y==0])
            loss = loss_one*self.pi + loss_zero*(1-self.pi)
        else:
            loss = self.criteria(score, Y)
          
        full_loss = loss
        if self.autoencoder > 0:
            full_loss = full_loss + recon_error*self.autoencoder
        full_loss.backward()

        p_hat = torch.sigmoid(score)
        precision = p_hat[Y == 1].sum().item()/p_hat.sum().item()
        tpr = p_hat[Y == 1].mean().item()
        fpr = p_hat[Y == 0].mean().item()

        if self.l2 > 0:
            r = sum(torch.sum(w**2) for w in self.model.features.parameters())
            r = r + sum(torch.sum(w**2) for w in self.model.classifier.parameters())
            r = 0.5*self.l2*r
            r.backward()

        self.optim.step()
        self.optim.zero_grad()

        if self.autoencoder > 0:
            return loss.item(), recon_error.item(), precision, tpr, fpr
        return (loss.item(),precision,tpr,fpr)


class GE_binomial:
    def __init__(self, model, optim, criteria, pi, l2=0
                , slack=1.0 #, labeled_fraction=0
                , entropy_penalty=0
                , autoencoder=0
                , posterior_L1=0):
        self.model = model
        self.optim = optim
        self.criteria = criteria
        self.slack = slack
        self.pi = pi # expectation of unlabeled only - do not include labeled positives
        self.entropy_penalty = entropy_penalty
        #self.labeled_fraction = labeled_fraction
        self.l2 = l2
        self.autoencoder = autoencoder
        self.posterior_L1 = posterior_L1

        self.header = ['loss', 'ge_penalty', 'precision', 'adjusted_precision', 'tpr', 'fpr']
        if self.autoencoder > 0:
            self.header = ['loss', 'ge_penalty', 'recon_error', 'precision', 'adjusted_precision', 'tpr', 'fpr']

    def step(self, X, Y):

        if self.autoencoder > 0:
            recon_error, score = autoencoder_loss(self.model, X)
        else:
            score = self.model(X).view(-1)

        select = (Y.data == 1)
        classifier_loss = self.criteria(score[select], Y[select])

        ## calculate Normal approximation to the distribution over positive count given
        ## by the classifier
        select = (Y.data == 0)
        N = select.sum().item()
        p_hat = torch.sigmoid(score[select])
        q_mu = p_hat.sum()  # expected number of positives according to classifier in whole image
        q_var = torch.sum(p_hat*(1-p_hat))  # linearity of variance of independent variables

        count_vector = torch.arange(0,N+1).float()
        count_vector = count_vector.to(q_mu.device)


        q_discrete = -0.5*(q_mu-count_vector)**2/(q_var + 1e-10) # add small epsilon to prevent NaN
        q_discrete = F.softmax(q_discrete, dim=0)

        ## KL of w from the binomial distribution with pi  (KL of classifier given binomial/pi)
        log_binom = scipy.stats.binom.logpmf(np.arange(0,N+1),N,self.pi)
        log_binom = torch.from_numpy(log_binom).float()

        ## Gaussian approximation of binomial prior - for the sake of testing 
        ## if Gaussian approximation compromises model performance
        # gaussian_approx = False
        # KL_only = True
        # if gaussian_approx:
        #     # Gaussian log PDF approximation
        #     log_gauss = scipy.stats.norm.logpdf(np.arange(0, N + 1), loc=N*self.pi, scale=np.sqrt(N * self.pi * (1 - self.pi)))
        #     log_binom = torch.from_numpy(log_gauss).float()

        if q_var.is_cuda:
            log_binom = log_binom.cuda()
        log_binom = Variable(log_binom)


        ge_penalty = -torch.sum(log_binom*q_discrete)  # cross entropy

        if self.entropy_penalty > 0:   # original code; see https://statproofbook.github.io/P/norm-dent.html
            q_entropy = 0.5*(torch.log(q_var) + np.log(2*np.pi) + 1)
            ge_penalty = ge_penalty + q_entropy*self.entropy_penalty

        if self.entropy_penalty < 0:   # get rid of gaussian/classifier's entropy term (if penalty=-1) -> KL-divergence!
            q_entropy = 0.5*(torch.log(q_var) + np.log(2*np.pi) + 1)
            ge_penalty = ge_penalty + q_entropy*self.entropy_penalty
            
        loss = classifier_loss + self.slack*ge_penalty

        if self.autoencoder > 0:
            loss = loss + recon_error*self.autoencoder

        if self.posterior_L1 > 0:
            r_labeled = torch.mean(torch.abs(score[Y==1]))
            r_unlabeled = torch.mean(torch.abs(score[Y==0]))
            r = self.posterior_L1*(r_labeled*self.labeled_fraction + r_unlabeled*(1-self.labeled_fraction))
            loss = loss + r

        loss.backward()

        p_hat = torch.sigmoid(score)
        precision = p_hat[Y == 1].sum().item()/p_hat.sum().item()
        tpr = p_hat[Y == 1].mean().item()
        fpr = p_hat[Y == 0].mean().item()

        if self.l2 > 0:
            r = sum(torch.sum(w**2) for w in self.model.features.parameters())
            r = r + sum(torch.sum(w**2) for w in self.model.classifier.parameters())
            r = 0.5*self.l2*r
            r.backward()

        self.optim.step()
        self.optim.zero_grad()

        if self.autoencoder > 0:
            return classifier_loss.item(), ge_penalty.item(), recon_error.item(), precision, tpr, fpr


class GE_KL:
    def __init__(self, model, optim, criteria, pi, l2=0
                , slack=1.0, momentum=1.0 #, labeled_fraction=0
                , entropy_penalty=0):
        self.model = model
        self.optim = optim
        self.criteria = criteria
        self.pi = pi
        self.l2 = l2
        self.slack = slack
        self.momentum = momentum
        self.running_expectation = pi
        #self.labeled_fraction = labeled_fraction
        self.entropy_penalty = entropy_penalty

        self.header = ['loss', 'ge_penalty', 'precision', 'adjusted_precision', 'tpr', 'fpr']

    def step(self, X, Y):

        X = Variable(X)
        Y = Variable(Y)

        score = self.model(X).view(-1)

        select = (Y.data == 1)
        classifier_loss = self.criteria(score[select], Y[select])

        select = (Y.data == 0)
        p_hat = torch.sigmoid(score[select]).mean()

        ## if labeled_fraction is > 0 then we are using positives in calculating the sample expectation
        #if self.labeled_fraction > 0:
        #    select = (Y.data == 1)
        #    p_label = p_hat.data.new(1).fill_(self.labeled_fraction)
        #    p_label = Variable(p_label, requires_grad=False)
        #    p_hat = (1.0-p_label)*p_hat + p_label*torch.sigmoid(score[select]).mean()

        ## p_hat is the expectation of the classifier over the data estimated from this minibatch
        ## if momentum is < 1 we are using an exponential running average of this quantity
        ## to include estimates from past minibatches
        if self.momentum < 1:
            p_hat = self.momentum*p_hat + (1-self.momentum)*self.running_expectation
            self.running_expectation = p_hat.item()

        entropy = self.pi*np.log(self.pi) + (1-self.pi)*np.log1p(-self.pi)
        ge_penalty = -torch.log(p_hat)*self.pi - torch.log1p(-p_hat)*(1-self.pi) + entropy 
        ge_penalty = ge_penalty*self.slack/self.momentum # divide by momentum to not change minibatch gradient magnitude 
    
        entropy_loss = 0
        if self.entropy_penalty > 0:
            select = (Y.data == 0)
            #p_hat = torch.sigmoid(score)
            #sign = (score > self.pi).float().detach()
            #entropy_loss = self.criteria(score[select], sign[select])

            ## penalize the entropy of the unlabeled data
            abs_score = torch.abs(score)
            log_p = F.logsigmoid(abs_score)
            one_minus_p = torch.sigmoid(-abs_score)
            entropy = abs_score*one_minus_p - log_p
            #log_one_minus_p = F.logsigmoid(-score)
            #p_hat = torch.exp(log_p)
            #one_minus_p_hat = torch.exp(log_one_minus_p)

            #entropy = -p_hat*log_p - one_minus_p_hat*log_one_minus_p
            #entropy[p_hat==0] = 0
            #entropy[one_minus_p_hat==1] = 0
            entropy_loss = self.entropy_penalty*entropy[select].mean()

      
        loss = classifier_loss + ge_penalty + entropy_loss
        loss.backward()

        p_hat = torch.sigmoid(score)
        precision = p_hat[Y == 1].sum().item()/p_hat.sum().item()
        tpr = p_hat[Y == 1].mean().item()
        fpr = p_hat[Y == 0].mean().item()

        if self.l2 > 0:
            r = 0.5*self.l2*sum(torch.sum(w**2) for w in self.model.parameters())
            r.backward()

        self.optim.step()
        self.optim.zero_grad()

        return classifier_loss.item(), ge_penalty.item(), precision, tpr, fpr


class PU:
    def __init__(self, model, optim, criteria, pi, l2=0
                , beta=0.0, autoencoder=0):
        # when beta = 0, this is NNPU
        self.model = model
        self.optim = optim
        self.criteria = criteria
        self.pi = pi
        self.l2 = l2
        self.beta = beta
        self.autoencoder = autoencoder

        self.header = ['loss', 'precision', 'adjusted_precision', 'tpr', 'fpr']
        if self.autoencoder > 0:
            self.header = ['loss', 'recon_error', 'precision', 'adjusted_precision', 'tpr', 'fpr']

    def step(self, X, Y):

        X = Variable(X)
        Y = Variable(Y)

        if self.autoencoder > 0:
            recon_error, score = autoencoder_loss(self.model, X)
        else:
            score = self.model(X).view(-1)

        loss_pp = self.criteria(score[Y==1], Y[Y==1])
        loss_pn = self.criteria(score[Y==1], 0*Y[Y==1]) # estimate loss for calling positives negative
        loss_un = self.criteria(score[Y==0], Y[Y==0])

        loss_u = loss_un - loss_pn*self.pi # estimate loss for negative data in unlabeled set
        if loss_u.item() < -self.beta:
            ## clip loss_u as in NNPU method https://arxiv.org/pdf/1703.00593.pdf
            ## in that paper they recommend taking a gradient step in the
            ## -loss_u direction in this case
            loss = -loss_u
            backprop_loss = loss
            loss_u = -self.beta
            loss = loss_pp*self.pi + loss_u
        else:
            loss = loss_pp*self.pi + loss_u
            backprop_loss = loss

        if self.autoencoder > 0:
            backprop_loss = backprop_loss + recon_error*self.autoencoder
        backprop_loss.backward()

        p_hat = torch.sigmoid(score)
        precision = p_hat[Y == 1].sum().item()/p_hat.sum().item()
        tpr = p_hat[Y == 1].mean().item()
        fpr = p_hat[Y == 0].mean().item()

        if self.l2 > 0:
            r = sum(torch.sum(w**2) for w in self.model.features.parameters())
            r = r + sum(torch.sum(w**2) for w in self.model.classifier.parameters())
            r = 0.5*self.l2*r
            r.backward()

        self.optim.step()
        self.optim.zero_grad()

        if self.autoencoder > 0:
            return loss.item(), recon_error.item(), precision, tpr, fpr

        return (loss.item(),precision,tpr,fpr)

class GE_multinomial:
    ''' implemented by Katie, Jan 2025 '''
    def __init__(self, model, optim, criteria, pi, l2=0
                , slack=1.0 #, labeled_fraction=0
                , entropy_penalty=0
                , autoencoder=0
                , posterior_L1=0):
        self.model = model
        self.optim = optim
        self.criteria = criteria  # should be nn.CrossEntropyLoss (categorical cross entropy, apt for multi-class)
        self.slack = slack
        pi = torch.tensor((pi,), dtype=torch.float32) if np.isscalar(pi) else pi
        self.pi = pi # expected frequencies of each particle type in unlabeled region; should not include particles from the labeled region
        self.entropy_penalty = entropy_penalty
        #self.labeled_fraction = labeled_fraction
        self.l2 = l2
        self.autoencoder = autoencoder
        self.posterior_L1 = posterior_L1

        self.header = ['loss', 'ge_penalty', 'precision', 'adjusted_precision', 'tpr', 'fpr']
        if self.autoencoder > 0:
            self.header = ['loss', 'ge_penalty', 'recon_error', 'precision', 'adjusted_precision', 'tpr', 'fpr']

        ## compute Sigma_2, the covariance matrix of class frequencies, as well as its inverse and determinant
        self.Sigma_2 = -torch.outer(pi, pi)  
        self.Sigma_2[range(len(pi)), range(len(pi))] = pi*(1-pi)  # Binomial variance down diagonal
        prob_neg = 1 - pi.sum()
        self.Sigma_2_det = (torch.prod(pi)*prob_neg).item()  # torch.det(self.Sigma_2).item()
        self.Sigma_2_inverse = 1/prob_neg * torch.ones(*self.Sigma_2.shape)  # torch.linalg.inv(self.Sigma_2)
        self.Sigma_2_inverse[range(len(self.Sigma_2)), range(len(self.Sigma_2))] = 1/pi + 1/prob_neg
        self.log_2pe = np.log(2*np.pi*np.e)/2 # constant to ease Gaussian entropy calculation later

    def step(self, X, Y):
        ''' 
        X is the batch with features
        Y is the class indices of each particle type; the first index (0) corresponds to the unlabeled regions), similar to GE-binomial
        '''
        if self.autoencoder > 0:
            recon_error, score = autoencoder_loss(self.model, X)
        else:
            score = self.model(X).squeeze()

        ## select data in the labeled region and calculate classifier loss for predicting particle types
        ## score is a 2darray with each row representing a vector of scores for each particle type

        ## first, convert scores to probabilities
        if score.ndim == 1:  # special case: binomial
            probs = torch.sigmoid(score)
            probs = torch.column_stack((1-probs, probs))  # prepend probability of not getting a particle
        else:
            probs = F.softmax(score, -1)

        Y = Y.to(torch.long)  # make sure Y is integer index
        select = (Y.data > 0)
        
        classifier_loss = self.criteria(probs[select], Y[select])

        ## calculate GE penalty as divergence of empirical distribution and multinomial distribution/prior
        p_hat = probs[~select, 1:]  # discard junk column (0th) and consider rows/regions that are unlabeled (not select)
        
        Sigma_1 = -torch.einsum('pi,pj->ij',p_hat,p_hat)/p_hat.shape[0]     # this is on the correct device
        Sigma_1[range(Sigma_1.shape[0]),range(Sigma_1.shape[1])] = (p_hat*(1-p_hat)).mean(0)
        Sigma_1_det = torch.det(Sigma_1)
        mu_diff = p_hat.mean(0).cpu() - self.pi.cpu()

        ## make sure everything is on the correct device
        mu_diff = mu_diff.to(p_hat.device)
        self.Sigma_2 = self.Sigma_2.to(p_hat.device)
        self.Sigma_2_inverse = self.Sigma_2_inverse.to(p_hat.device)
        self.pi = self.pi.to(p_hat.device)

        
        ## KL divergence of two multivariate normals - https://statproofbook.github.io/P/mvn-kl.html
        ge_penalty = 0.5*((self.Sigma_2_inverse @ Sigma_1).trace() + \
                mu_diff @ self.Sigma_2_inverse @ mu_diff - \
                Sigma_1.shape[0] + \
                torch.log(self.Sigma_2_det/torch.det(Sigma_1)))

        x_entropy = True    # entropy of classifier multivariate gaussian - https://statproofbook.github.io/P/mvn-dent.html
        if x_entropy:   # include entropy of prior/multinomial (approx w/ gaussian), so sum is now cross entropy
            entropy = p_hat.shape[1]*self.log_2pe + 0.5*torch.log(Sigma_1_det)
            ge_penalty += entropy
            
        loss = classifier_loss + self.slack * ge_penalty

        if self.autoencoder > 0:
            loss += recon_error * self.autoencoder

        if self.posterior_L1 > 0:
            r_labeled = torch.mean(torch.abs(score[Y>0]))
            r_unlabeled = torch.mean(torch.abs(score[Y==0]))
            r = self.posterior_L1*(r_labeled*self.labeled_fraction + r_unlabeled*(1-self.labeled_fraction))
            loss = loss + r
        torch.autograd.set_detect_anomaly(True)   # break if any NaNs arise
        loss.backward()

        precision = (probs[Y>0, 1:].sum()/probs[:,1:].sum()).item()
        tpr = probs[Y>0, 1:].mean().item()
        fpr = probs[Y==0, 1:].mean().item()

        if self.l2 > 0:
            r = sum(torch.sum(w**2) for w in self.model.features.parameters())
            r = r + sum(torch.sum(w**2) for w in self.model.classifier.parameters())
            r = 0.5*self.l2*r
            r.backward()

        self.optim.step()
        self.optim.zero_grad()

        if self.autoencoder > 0:
            return classifier_loss.item(), ge_penalty.item(), recon_error.item(), precision, tpr, fpr

        return classifier_loss.item(), ge_penalty.item(), precision, tpr, fpr
