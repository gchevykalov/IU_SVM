import numpy as np

EPS = 0.00001
INF = 1e12

class IUSVMClassifier:
    def __init__(self, kernel, C):
        self.alphas = []
        self.mu = 0
        self.x = []
        self.y = []
        self.kernel = kernel

        self.C = C
        self.Q = np.array([[0]], np.float32)
        self.g = []

        self.S = []
        self.E = []
        self.O = []


    def deleteFromSetS(self, index):
        k = self.S.index(index) + 1
        q_12 = np.delete(self.Q[:,k].reshape(-1, 1), k, 0)
        q_21 = np.delete(self.Q[k,:].reshape(1, -1), k, 1)
        q_22 = self.Q[k,k]
        q_11 = np.delete(np.delete(self.Q, k, 0), k, 1)
        self.Q = q_11 - q_12 @ q_21 / q_22
        self.S.remove(index)


    def addToSetS(self, index):
        nu = np.array([[self.y[index]] +
                      [self.y[s] * self.y[index] * self.kernel(self.x[index], self.x[s]) for s in self.S]],
                      np.float32).T
        betas = np.append(-1 * self.Q @ nu, [[1]], 0)
        k = self.y[index] * self.y[index] * self.kernel(self.x[index], self.x[index]) - nu.T @ self.Q @ nu
        self.Q = np.append(
            np.append(self.Q, np.array([[0] for i in range(len(self.S) + 1)]), 1),
            np.array([[0 for i in range(len(self.S) + 2)]]),
            0
        ) + 1 / k * betas @ betas.T
        self.S.append(index)


    def calculateBetas(self):
        nu = np.array([[self.y[-1]] +
                      [self.y[s] * self.y[-1] * self.kernel(self.x[-1], self.x[s]) for s in self.S]],
                      np.float32).T
        return -1 * self.Q @ nu


    def calculateGammas(self, betas):
        K_cc_cr = np.array([[self.y[r] * self.y[-1] * self.kernel(self.x[-1], self.x[r])] for r in [-1] + self.E + self.O],
                           np.float32)
        m = np.concatenate([
            [[self.y[r]] for r in [-1] + self.E + self.O],
            [[self.y[r] * self.y[s] * self.kernel(self.x[r], self.x[s]) for s in self.S] for r in [-1] + self.E + self.O]],
            axis=1,
            dtype=np.float32
        )
        return m @ betas + K_cc_cr


    def calculateDelta(self, betas, gammas):
        k1 = None
        k2 = None
        k3 = None

        deltaAmax = []
        for i in range(1, len(self.S) + 1):
            if betas[i] > EPS:
                deltaAmax.append((self.C - self.alphas[self.S[i - 1]]) / betas[i])
            elif betas[i] < -EPS:
                deltaAmax.append(-self.alphas[self.S[i - 1]] / betas[i])

        if len(deltaAmax) != 0:
            i = np.argmin(deltaAmax)
            sign = 1 if deltaAmax[i] >= 0 else 0
            newArr = [abs(x_i) * sign for x_i in deltaAmax]
            k1 = self.S[np.argmin(newArr)]
            deltaAcS = np.min(newArr)
        else:
            deltaAcS = INF

        deltas = []
        for i in range(len(self.E)):
            if gammas[i + 1] > EPS:
                deltas.append(-self.g[self.E[i]] / gammas[i + 1])
        shift = len(self.E)
        for i in range(len(self.O)):
            if gammas[i + 1 + shift] < -EPS:
                deltas.append(-self.g[self.O[i]] / gammas[i + 1 + shift])

        if len(deltas) != 0:
            deltaAcR = min(deltas)
            i = np.argmin(deltas)
            if i < shift:
                k2 = self.E[i]
            else:
                k2 = self.O[i - shift]
        else:
            deltaAcR = INF

        if gammas[0] > EPS:
            deltaAcg = -self.g[-1] / gammas[0]
        else:
            deltaAcg = INF

        deltaAca = self.C - self.alphas[-1]
        k3 = len(self.alphas) - 1

        deltaAmaxC = min([deltaAcS, deltaAcR, deltaAcg, deltaAca])
        if deltaAmaxC == deltaAcS:
            return deltaAmaxC, k1
        elif deltaAmaxC == deltaAcR:
            return deltaAmaxC, k2
        else:
            return deltaAmaxC, k3


    def processEmptySetS(self):
        for e in self.E:
            if abs(self.g[e]) < EPS:
                self.addToSetS(e)
                self.E.remove(e)
                return False

        for o in self.O:
            if abs(self.g[o]) < EPS:
                self.addToSetS(o)
                self.O.remove(o)
                return False

        mus_r = []
        for i in range(len(self.alphas)):
            if i in self.E and -(self.y[i] / self.y[-1]) > EPS:
                mus_r.append(-self.g[i] / self.y[i])
            elif i in self.O and -(self.y[i] / self.y[-1]) < -EPS:
                mus_r.append(-self.g[i] / self.y[i])
        if len(mus_r) != 0:
            mu_r = min(mus_r)
        else:
            mu_r = INF
        mu_c = -self.g[-1] / self.y[-1]
        mu = min(mu_r, mu_c)

        for i in range(len(self.alphas)):
            self.g[i] += self.y[i] * mu

        for e in self.E:
            if abs(self.g[e]) < EPS:
                self.addToSetS(e)
                self.E.remove(e)
                return False

        for o in self.O:
            if abs(self.g[o]) < EPS:
                self.addToSetS(o)
                self.O.remove(o)
                return False

        self.addToSetS(len(self.alphas) - 1)
        self.mu += mu
        return True


    def incrementPair(self, x_c, y_c):
        alpha_c = 0
        g_c = y_c * self(x_c) - 1

        self.x.append(x_c)
        self.y.append(y_c)
        self.alphas.append(alpha_c)
        self.g.append(g_c)

        while self.g[-1] < -EPS and self.alphas[-1] < self.C:
            if len(self.S) == 0:
                isNeedToLeave = self.processEmptySetS()
                if isNeedToLeave:
                    return

            betas = self.calculateBetas()
            gammas = self.calculateGammas(betas)

            delta_ac, k = self.calculateDelta(betas, gammas)

            self.mu += betas[0] * delta_ac
            for i in range(len(self.S)):
                self.alphas[self.S[i]] += betas[i + 1] * delta_ac
            self.alphas[-1] += delta_ac
            for i in range(len(self.E)):
                self.g[self.E[i]] += gammas[i + 1] * delta_ac
            shift = len(self.E)
            for i in range(len(self.O)):
                self.g[self.O[i]] += gammas[i + 1 + shift] * delta_ac
            self.g[-1] += gammas[0] * delta_ac

            if k in self.S:
                self.deleteFromSetS(k)
                if abs(self.alphas[k]) < EPS:
                    self.O.append(k)
                else:
                    self.E.append(k)
            elif k in self.O:
                self.addToSetS(k)
                self.O.remove(k)
            elif k in self.E:
                self.addToSetS(k)
                self.E.remove(k)
            else:
                if abs(self.g[-1]) < EPS:
                    self.addToSetS(k)
                else:
                    self.E.append(k)
                break

        new_index = len(self.x) - 1
        if new_index not in self.S and new_index not in self.E:
            self.O.append(new_index)


    def __call__(self, x):
        res = self.mu

        for i in range(len(self.alphas)):
            res += self.alphas[i] * self.y[i] * self.kernel(self.x[i], x)

        return res