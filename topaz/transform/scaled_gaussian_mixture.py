from __future__ import print_function, division

import numpy as np

class ScaledGaussianMixture:

    def __init__(self, ncomponents=2, scale_prior=0.5, tol=1e-6):
        self.ncomponents = ncomponents
        self.weights = np.ones(ncomponents)/ncomponents
        self.means = np.zeros(ncomponents)
        self.variances = np.ones(ncomponents)
        self.scale_prior = scale_prior
        self.tol = tol

    def fit(self, X, niters=100, random=np.random, verbose=False):

        X = [X[i].ravel() for i in range(len(X))]

        weights = self.weights
        means = self.means
        variances = self.variances

        mus = np.array([np.mean(X[i]) for i in range(len(X))]) #, dtype=np.float32)
        scale = mus/np.mean(mus)

        probas = []

        for i in range(len(X)):
            component = random.randint(0, self.ncomponents, size=X[i].shape).astype(np.int32)
            size = (len(X[i]), self.ncomponents)
            proba = np.zeros(size, dtype=np.float32)
            proba[np.arange(len(X[i])), component] = 1.0
            probas.append(proba)

            #proba = np.ones(size, dtype=np.float32)
            #proba = np.random.uniform(size=size).astype(np.float32)
            #proba /= proba.sum(axis=-1, keepdims=True)

            #component = proba.argmax(axis=-1).astype(np.int32)
            #components.append(component)

        n = np.zeros(self.ncomponents, dtype=np.float64)

        logp = -np.inf

        ## coordinate ascent iteration
        for it in range(niters):

            #print('maximize')
            ## compute maximum likelihood model parameters
            means[:] = 0
            variances[:] = 0
            n[:] = 0

            for i in range(len(X)):
                proba = probas[i]
                xi = X[i]/scale[i]

                n += proba.sum(0)
                delta = xi[...,np.newaxis] - means
                means += np.sum(proba*delta, axis=0)/n
                delta2 = xi[...,np.newaxis] - means
                variances += np.sum(delta*delta2*proba, axis=0)
                """
                for j in range(self.ncomponents):
                    n[j] += np.sum(proba[:,j])
                    delta = xi - means[j]
                    means[j] += np.sum(proba[:,j]*delta)/n[j]
                    delta2 = xi - means[j]
                    variances[j] += np.sum(proba[:,j]*delta*delta2)
                """


            variances /= n
            #weights[:] = (n+1)/(np.sum(n) + self.ncomponents)
            weights[:] = (n+1)
            weights /= np.sum(n) + self.ncomponents

            #print(weights)
            #print(means)
            #print(variances)

            #print('expect')

            ## calculate the expectation of the scaling parameters and mixture components
            cur_logp = logp
            logp = 0

            for i in range(len(X)):
                # scale parameters
                proba = probas[i]
                component = proba.argmax(axis=-1).astype(np.int32)
                xi = X[i]

                a = np.sum(proba*xi[...,np.newaxis]**2/variances)
                b = np.sum(proba*xi[...,np.newaxis]*means/variances)
                scale[i] = 2*a/(b + np.sqrt(b**2 + 4*a*len(xi)))

                # mixture components
                cur_proba = proba
                next_proba = -(xi[...,np.newaxis]/scale[i]-means)**2/2/variances - np.log(2*np.pi)/2 - np.log(variances)/2
                next_proba += np.log(weights)

                ma = next_proba.max(axis=-1, keepdims=True)
                next_proba -= ma

                logp += np.sum(np.log(np.sum(np.exp(next_proba), axis=-1))) + np.sum(ma)

                next_proba[:] = np.exp(next_proba)
                next_proba /= next_proba.sum(axis=-1, keepdims=True)
                probas[i] = next_proba

            if verbose:
                digits = int(np.floor(np.log10(niters))) + 1
                template = '# [{:0'+str(digits)+'d}] logp={}'
                print(template.format(it, logp))

            if logp - cur_logp < self.tol: # stop, parameters have converged
                print('# logp tolerance reached')
                break

        return scale, probas


    def transform(self, X, niters=5):

        weights = self.weights
        means = self.means
        variances = self.variances

        mus = np.array([np.mean(X[i]) for i in range(len(X))], dtype=np.float32)
        scale = np.mean(mus)/mus
        #cdef np.ndarray[float] scale = np.ones(len(X), dtype=np.float32)

        components = []
        probas = []
        for i in range(len(X)):
            component = np.random.randint(0, self.ncomponents, size=X[i].shape).astype(np.int32)
            components.append(component)
            #size = (X[i].shape[0], X[i].shape[1], self.ncomponents)
            #proba = np.ones(size, dtype=np.float32)
            #proba += np.random.uniform(0,0.1,size=size).astype(np.float32)
        
        # mixture components
        for i in range(len(X)):
            component = components[i]
            xi = X[i]
            proba = np.exp(-(xi[...,np.newaxis]/scale[i]-means)**2/2/variances)/np.sqrt(2*np.pi*variances)
            proba *= weights
            component[:,:] = proba.argmax(axis=-1).astype(np.int32)

        for _ in range(niters):
            ## calculate the expectation of the scaling parameters and mixture components
            for i in range(len(X)):
                # scale parameters
                a = 0
                b = 0
                component = components[i]
                xi = X[i]

                mu = means[component]
                var = variances[component]

                a = np.sum(xi**2/var)
                b = np.sum(xi*mu/var)
                scale[i] = a/b

                unscale_logp = np.log(1 - self.scale_prior) - np.sum((xi-mu)**2/2/var)
                scale_logp = np.log(self.scale_prior) - np.sum((xi/scale[i]-mu)**2/2/var)
                if unscale_logp >= scale_logp:
                    scale[i] = 1.0

                # mixture components
                proba = np.exp(-(xi[...,np.newaxis]/scale[i]-means)**2/2/variances)/np.sqrt(2*np.pi*variances)
                proba *= weights
                component[:,:] = proba.argmax(axis=-1).astype(np.int32)


        return scale, proba



                







        
        


