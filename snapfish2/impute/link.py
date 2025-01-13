import numpy as np
from statsmodels.api import GLM, families, OLS

class LinkHiCFISH:
    def __init__(self, arr, hic_mat, b_init=None, tau_init=None):
        self.pdists = np.nanmean(np.abs(arr), axis=0)
        self.hic_mat = hic_mat
        
        if b_init is None:
            self.beta = -1
        else:
            self.beta = b_init
            
        if tau_init is None:
            self.tau_sq = np.ones(arr.shape[1], dtype="float64")
            
        self.fit_poisson()
        self.fit_ols()
            
    def fit_poisson(self):
        d, p = self.pdists.shape[:2]
        uidx = np.triu_indices(p, 1)
        X = np.concatenate(self.pdists[:,*uidx])
        dummy = np.zeros((d, len(X)))
        for i in range(d):
            dummy[i,i*len(uidx[0]):(i+1)*len(uidx[0])] = 2
        X = np.vstack([dummy, X]).T
        
        y = np.tile(self.hic_mat[uidx], d)
        
        self.glm = GLM(exog=X, endog=y, family=families.Poisson()).fit()
        
    def fit_ols(self):
        d, p = self.pdists.shape[:2]
        uidx = np.triu_indices(p, 1)
        hic_positive = np.where(self.hic_mat == 0, 0.5, self.hic_mat)
        X = np.log(np.tile(hic_positive[uidx], d))
        dummy = np.zeros((d, len(X)))
        for i in range(d):
            dummy[i,i*len(uidx[0]):(i+1)*len(uidx[0])] = 2**.5
        X = np.vstack([dummy, X]).T
        
        y = np.concatenate(self.pdists[:,*uidx])
        
        self.ols = OLS(exog=X, endog=y).fit()
        
    
    def fit_ols_rev(self):
        d, p = self.pdists.shape[:2]
        uidx = np.triu_indices(p, 1)
        hic_positive = np.where(self.hic_mat == 0, 0.5, self.hic_mat)
        X = np.log(np.tile(hic_positive[uidx], d))**2
        dummy = np.zeros((d, len(X)))
        for i in range(d):
            dummy[i,i*len(uidx[0]):(i+1)*len(uidx[0])] = 2
        X = np.vstack([dummy, X]).T
        
        y = np.concatenate(self.pdists[:,*uidx])**2
        
        # Not Gaussian error
        # self.ols = OLS(exog=X, endog=y).fit()
        
        beta = np.linalg.inv(X.T@X)@X.T@y
        return beta
        
            
    @property
    def params(self):
        return np.concatenate([[self.beta], self.tau_sq])
        
    def loglklhd(self, beta=None, tau_sq=None):
        uidx = np.triu_indices(self.pdists.shape[-1], 1)
        h = self.hic_mat[uidx]
        if beta is None:
            beta = self.beta
        if tau_sq is None:
            tau_sq = self.tau_sq
        part = np.stack([
            (np.pi*np.square(t1)/2 - 2*t2)[uidx]
            for t1, t2 in zip(self.pdists, tau_sq)
        ])
        return np.sum(beta*h*np.sqrt(part) - np.exp(np.sqrt(part)))
        
    def _update(self):
        uidx = np.triu_indices(self.pdists.shape[-1], 1)
        h = self.hic_mat[uidx]
        part = np.stack([
            (np.pi*np.square(t1)/2 - 2*t2)[uidx]
            for t1, t2 in zip(self.pdists, self.tau_sq)
        ])
        p1_ijj_beta = h*np.sqrt(part)
        p1_ijj_taui = (part)**(-.5)*(np.exp(np.sqrt(part)) - h*self.beta)
        p1_beta = np.sum(p1_ijj_beta)
        p1_taui = np.sum(p1_ijj_taui, axis=1)
        p1 = np.concatenate([[p1_beta], p1_taui])
        print(p1)

        p2_beta_taui = np.sum(-h*part**(-.5), axis=1)
        p2_ijj_taui_taui = (p1_ijj_taui + np.exp(np.sqrt(part)))/part
        p2_taui_taui = np.sum(p2_ijj_taui_taui, axis=1)
        
        M2 = np.diag(p2_taui_taui)
        m12 = p2_beta_taui[:,None]
        r1 = np.concatenate([[0], m12.T[0]])[None,:]
        r2 = np.hstack([m12, M2])
        Mi = np.linalg.inv(np.vstack([r1, r2]))

        # M2i = np.diag(1/p2_taui_taui)
        # m12 = p2_beta_taui[:,None]
        # Si = -(1/m12.T@M2i@m12)[0][0]
        # a12 = -Si*m12.T@M2i
        # a22 = M2i@m12@m12.T@M2i*Si + M2i
        # r1 = np.concatenate([[Si], a12[0]])[None,:]
        # r2 = np.hstack([a12.T, a22])
        # Mi = np.vstack([r1, r2])

        change = Mi@p1
        self.beta = self.beta - change[0]
        self.tau_sq = self.tau_sq - change[1:]

    def _ascent(self):
        uidx = np.triu_indices(self.pdists.shape[-1], 1)
        h = self.hic_mat[uidx]
        part = np.stack([
            (np.pi*np.square(t1)/2 - 2*t2)[uidx]
            for t1, t2 in zip(self.pdists, self.tau_sq)
        ])
        p1_ijj_beta = h*np.sqrt(part)
        p1_ijj_taui = (part)**(-.5)*(np.exp(np.sqrt(part)) - h*self.beta)
        p1_beta = np.sum(p1_ijj_beta)
        p1_taui = np.sum(p1_ijj_taui, axis=1)
        p1 = np.concatenate([[p1_beta], p1_taui])
        print(p1)
        
        if p1[0] < 0:
            self.beta -= 0.01
        else:
            self.beta += 0.01
            
        for i, t in enumerate(p1[1:]):
            if t < 0:
                self.tau_sq[i] -= 10
            else:
                self.tau_sq[i] += 10