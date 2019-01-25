# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:00:27 2019

@author: Felix
"""

import numpy as np
# import calcSpline
import scipy.integrate

ODEsolver = scipy.integrate.odeint


class parameters:
    def __init__(self, m, Iz, lv, lh, EG):
        self.m = m  # Masse in kg
        self.Iz = Iz  # Trägheitsmoment in kg*m**2
        self.lv = lv  # abstand von schwerpunkt zu vorderreifen in m
        self.lh = lh  # abstand von schwerpunkt zu hinterreifen in m
        self.L = lv + lh  # gesamtlänge in m
        self.EG = EG  # eigenlenkgradient aus parameterident
        self.ks = (lv - lh) / EG / self.L * m  # Schräglaaufsteiigkeit vorne und hinten


class virtualCarContainer:
    def __init__(self, x, v, delta):
        self.state = x
        self.v = v
        self.delta = delta


class invModelControl:
    def __init__(self, Vsoll, W, trajectoryType="cubicS", Vstart=None, accSoll=0.1, Psi=None, R=None):
        self.solveroptions = {'hmax': 0.1}
        m = 2.26113  # Masse in kg
        Iz = 0.0274  # Trägheitsmoment in kg*m**2
        lv = 0.16812  # abstand von schwerpunkt zu vorderreifen
        lh = 0.11687  # abstand von schwerpunkt zu hinterreifen
        EG = 0.0547  # eigenlenkgradient aus parameterident
        self.param = parameters(m, Iz, lv, lh, EG)
        self.Vsoll = Vsoll
        self.trajectory = trajectory(Vsoll, W, name=trajectoryType, Vstart=Vstart, Psi=Psi, R=R)
        self.overtake = False
        self.T0 = None
        self.accSoll = accSoll
        self.lastKnownFrontCar = virtualCarContainer(np.array([0, W, 0, 0, 0]), Vsoll, 0)

    ################
    # first all functions for the changelane maneuver
    ################

    def carInput(self, t, phase='changeLane', dpsi=None):
        if phase == 'changeLane':
            p, dp, ddp = self.trajectoryGen(t)
            v, delta, psi = self.invModel(p, dp, ddp)
            if delta > self.degToRad(29):
                delta = self.degToRad(29)
            elif delta < self.degToRad(-29):
                delta = self.degToRad(-29)
        elif phase == 'straightLine':
            v = self.Vsoll  # +self.accSoll*t
            delta = 0
            psi = 0
        #        elif phase == 'circleLine':
        #            p,dp,ddp = self.circle(t)
        #            v,delta,psi = self.invModel(p,dp,ddp)

        return v, delta, psi

    def circle(self, t):
        Rs = self.trajectory.specifics.R - self.trajectory.specifics.W
        dpsi = self.Vsoll / Rs
        psi = dpsi * t
        x = self.trajectory.specifics.R * np.sin(psi)
        y = -Rs * np.cos(psi)
        p = np.array([x, y])
        dx = self.trajectory.specifics.R * dpsi * np.cos(psi)
        dy = Rs * dpsi * np.sin(psi)
        dp = np.array([dx, dy])
        ddx = -self.trajectory.specifics.R * dpsi ** 2 * np.sin(psi)
        ddy = Rs * dpsi ** 2 * np.cos(psi)
        ddp = np.array([ddx, ddy])
        return p, dp, ddp

    def completeManeuver(self, t, maneuver='line'):
        T = self.trajectory.specifics.T
        delay = 1
        if maneuver == 'line':
            if 0 <= t <= T:
                # self.trajectory.setSpecifics([self.Vsoll,self.trajectory.specifics.W])
                v, delta, psi = self.carInput(t, phase='changeLane')
                # print(psi)
            elif T + delay < t <= 2 * T + delay:
                if not self.T0:
                    print("init")
                    self.T0 = t
                # self.trajectory.setSpecifics([self.Vsoll,-self.trajectory.specifics.W])
                v, delta, psi = self.carInput(t - self.T0, phase='changeLane')
                delta = -delta
                psi = -psi
            else:
                v, delta, psi = self.carInput(t, phase='straightLine')
        elif maneuver == 'circle':
            v, delta, psi = self.carInput(t, phase='circleLine')
        return v, delta, psi

    def trajectoryControler(self, error, t=None, p=10):
        maxerror = 30
        maxsteering = 29
        if not t:
            errorConv = error / maxerror * maxsteering
            return p * errorConv
        else:
            return self.funnelControler(-error, t, maxsteering)

    def funnelControler(self, error, t, a):
        '''
        parameters
        ---------
        error : = ist - soll ( das ist die negative Definition unserer bisherigen Fehler Definition)
        a : ist die Sättigung, was der Regler maximal an Lenkwinkel ändern kann.
            (beachte, dass wir den Output auf das delta der Vorsteuerung noch addieren)
        t : ist die Zeit. hier könnte man überlegen, ob man wie bei der Trajektorie wieder über die Referenzzeit von null startet.
        '''
        ph = self.funnel(t)
        if abs(ph * error) < 1:
            # print('inside of funnel')
            u = -error / (1 - ph ** 2 * error ** 2)
            u = self.saturate(u, a)
        else:
            # print('outside of funnel')
            u = -a * np.sign(error)
        return u

    def funnel(self, t):
        return np.e ** (-t) + 1

    def saturate(self, u, a):
        if u >= 0:
            u = np.min([u, a])
        else:
            u = np.max([u, -a])
        return u

    def invModel(self, p, dp, ddp):
        # dxsoll**2+dysoll**2 unequal 0
        # p is the desired trajectory dp ddp are derivative w.r.t. time
        dxsoll = dp[0]
        dysoll = dp[1]
        ddxsoll = ddp[0]
        ddysoll = ddp[1]
        psi = np.arctan2(dysoll, dxsoll)
        dpsi = (dxsoll * ddysoll - dysoll * ddxsoll) / (dxsoll ** 2 + dysoll ** 2)
        v = dxsoll * np.cos(psi) + dysoll * np.sin(psi)
        delta = np.arctan2(self.param.L * dpsi, v)
        return v, delta, psi

    def trajectoryGen(self, t):
        if self.trajectory.specifics.name == 'nineP':
            xsoll, ysoll = self.trajectory.generateParametrizedPath(t,None)
            dxsoll, dysoll = self.trajectory.generateParametrizedPath(t,xsoll, derivative='first')
            ddxsoll, ddysoll = self.trajectory.generateParametrizedPath(t,xsoll, derivative='second')
            p = np.array([xsoll, ysoll]).reshape(2)
            # modeling that y is a function of x(t) :  y(x(t)) leads to chainrule
            dp = np.array([dxsoll,dysoll * dxsoll]).reshape(2)
            ddp = np.array([ddxsoll,ddysoll * dxsoll**2 + ddxsoll * dysoll]).reshape(2)

        else:
            xsoll = self.trajectory.generateXtrajectory(t)  # self.Vsoll*t
            dxsoll = self.trajectory.generateXtrajectory(t, derivative='first')  # self.Vsoll
            ddxsoll = self.trajectory.generateXtrajectory(t, derivative='second')  # 0
            ysoll = self.trajectory.generateSpline(t)
            dysoll = self.trajectory.generateSpline(t, derivative='first')
            ddysoll = self.trajectory.generateSpline(t, derivative='second')

            p = np.array([xsoll, ysoll]).reshape(2)
            dp = np.array([dxsoll, dysoll]).reshape(2)
            ddp = np.array([ddxsoll, ddysoll]).reshape(2)
        return p, dp, ddp



    def radToDeg(self, x):
        return x / np.pi * 180

    def degToRad(self, x):
        return x * np.pi / 180

    ################
    # all functions for simulating while overtaking
    ################

    def simulateModel(self, y0, trange, model='complex', **kwargs):

        if model == 'complex':
            if y0.size < 5:
                y0 = np.concatenate((y0, np.zeros((5 - y0.size))), axis=None)
            try:
                ode = lambda x, t: self.carModelOneLane(x, t, **kwargs)
            except TypeError as error:
                print(error)
            y = ODEsolver(ode, y0, trange, **self.solveroptions)
        elif model == 'parallel':
            if y0.size < 10:
                y0 = np.concatenate((y0, np.zeros((10 - y0.size,))), axis=None)
            try:
                ode = lambda x, t: self.carModelParallel(x, t, **kwargs)
            except TypeError as error:
                print(error)

            y = ODEsolver(ode, y0, trange, **self.solveroptions)
        elif model == 'discrete':
            if y0.size < 10:
                y0 = np.concatenate((y0, np.zeros((10 - y0.size,))), axis=None)
            try:
                ode = lambda x, t: self.carModelDiscrete(x, t, **kwargs)
            except TypeError as error:
                print(error)

            step = trange / 5
            t = np.arange(0, trange + step, step)
            y = ODEsolver(ode, y0, t)

        elif model == 'full':
            if y0.size < 5:
                y0 = np.concatenate((y0, np.zeros((5 - y0.size,))), axis=None)
            try:
                ode = lambda x, t: self.carModelFull(x, t)
            except TypeError as error:
                print(error)

            y = ODEsolver(ode, y0, trange, **self.solveroptions)
        elif model == 'simple':
            y0 = y0[:3]
            try:
                ode = lambda x, t: self.carModelOneLaneSimple(x, t, **kwargs)
            except TypeError as error:
                print(error)
            y = ODEsolver(ode, y0, trange, **self.solveroptions)
        else:
            raise NameError('given model type does not match any implemented one')
        return y

    def carModelOneLaneSimple(self, x, t0, control=True, constInput=False, uc=[1, 0]):
        #        if 'control' in kwargs.keys():
        #            control = kwargs['control']
        #        else:
        #            control = False
        Xpos = x[0]
        Ypos = x[1]
        psi = x[2]
        if constInput:
            v = uc[0]
            delta = uc[1]
        else:
            v, delta, psisoll = self.carInput(t0)
            if control:
                error = self.radToDeg(psisoll - psi)
                ddelta = self.trajectoryControler(error)
                delta = delta + self.degToRad(ddelta)

        dx = np.array([np.cos(psi), np.sin(psi), np.tan(delta) / self.param.L]) * v
        return dx

    def carModelOneLane(self, x, t0, control=True, constInput=False, uc=[1, 0], completeManeuver=False):
        #        if 'constInput' in kwargs.keys():
        #            constInput = kwargs['constInput']
        #            if 'uc' in
        #        else:
        #            constInput = False
        # %% input and constants
        vTol = 1.2 * 10 ** (-3)  # toleranz for small velocitys to switch to simple model
        # %% states
        xpos = x[0]
        ypos = x[1]
        betha = x[2]
        psi = x[3]
        phi = x[4]  # phi is dummy for dpsi
        if constInput:
            v = uc[0]
            delta = uc[1]
        elif completeManeuver:
            v, delta, psisoll = self.completeManeuver(t0, maneuver='line')
            if control:
                error = self.radToDeg(psisoll - psi)
                ddelta = self.trajectoryControler(error, t=t0)
                delta = delta + self.degToRad(ddelta)
                delta = self.saturate(delta, self.degToRad(29))


        else:
            v, delta, psisoll = self.carInput(t0)
            if control:
                error = self.radToDeg(psisoll - psi)
                ddelta = self.trajectoryControler(error,t0)
                delta = delta + self.degToRad(ddelta)
                delta = self.saturate(delta, self.degToRad(29))
        if v == 0:
            v = vTol
        v = np.sign(v) * np.max([np.abs(v), vTol])
        if (np.abs(v) >= vTol):
            # %% forces (possible to incorporate  more)
            alpha_v = delta - betha - self.param.lv * phi / v
            alpha_h = self.param.lh * phi / v - betha
            Fv = self.param.ks * self.radToDeg(alpha_v)  # ks is in N/deg
            Fh = self.param.ks * self.radToDeg(alpha_h)
            Fx = -Fv * np.sin(delta)
            Fy = Fh + np.cos(delta) * Fv
            # %% equations
            dpsi = phi
            dbetha = (Fy * np.cos(betha) ** 2 - np.cos(betha) * np.sin(betha) * Fx) / (self.param.m * v) - dpsi
            dphi = (np.cos(delta) * Fv * self.param.lv - Fh * self.param.lh) / self.param.Iz
            dxpos = np.tan(betha) * np.sin(psi) * v + np.cos(psi) * v
            dypos = -np.tan(betha) * np.cos(psi) * v + np.sin(psi) * v
        else:
            smallx = np.array([xpos, ypos, psi])
            dx = self.carModelOneLaneSimple(smallx, t0)
            dxpos = dx[0]  # np.cos(psi)*v
            dypos = dx[1]  # np.sin(psi)*v
            dbetha = 0
            dpsi = dx[2]  # 0
            dphi = 0
        return [dxpos, dypos, dbetha, dpsi, dphi]

    def carModelDiscrete(self, xa, t0, ub=[1, 0], uf=[1, 0]):
        n = xa.size // 2
        dx1 = self.carModelOneLane(xa[:n], t0, constInput=True, uc=ub)
        dx2 = self.carModelOneLane(xa[n:], t0, constInput=True, uc=uf)
        return np.concatenate((dx1, dx2), axis=None)

    def carModelParallel(self, xa, t0, control=True, uc=[1, 0]):
        n = xa.size // 2
        psiFollow = xa[3]
        xFollow = xa[0]
        xFront = xa[5]
        if 0 <= t0 <= self.trajectory.specifics.T:
            v, delta, psiSoll = self.carInput(t0, phase='changeLane')

        else:
            if xFollow - self.param.lh >= xFront:
                if not self.T0:  # chaneLane Back started
                    self.T0 = t0
                    print('T0')
                    print(self.T0)
                    W = self.trajectory.specifics.W
                    self.trajectory.setSpecifics([self.Vsoll, -W])

                if t0 - self.T0 > 0:
                    print(t0 - self.T0)
                if 0 <= (t0 - self.T0) <= self.trajectory.specifics.T:
                    print('change back')
                    v, delta, psiSoll = self.carInput(t0 - self.T0, phase='changeLane')

                else:
                    v, delta, psiSoll = uc[0], uc[1], 0

            else:
                v, delta, psiSoll = self.carInput(t0 - self.trajectory.specifics.T, phase='straightLine')
                self.trajectory.updateVsoll(v)

        if control:
            error = self.radToDeg(psiSoll - psiFollow)
            ddelta = self.trajectoryControler(error)
            delta = delta + self.degToRad(ddelta)

        dx1 = self.carModelOneLane(xa[:n], t0, constInput=True, uc=[v, delta])
        dx2 = self.carModelOneLane(xa[n:], t0, constInput=True, uc=uc)
        return np.concatenate((dx1, dx2), axis=None)

    def carModelFull(self, x, t0):
        psi = x[3]
        # %% externaly call carInput depending on what the internal simulation says
        if not self.T0:  # T0 is only initialized once as None
            self.overtake = self.tryToOvertake(self.lastKnownFrontCar, x, t0)
            if self.overtake:
                self.T0 = t0  # save overtake starting time
                print(t0)
        if self.overtake == True:
            vc, deltac, psisoll = self.carInput(
                t0 - self.T0)  # get carInout due to trajectory change relative to start overtake T0
        else:
            vc, deltac, psisoll = self.carInput(t0, phase='straightLine')
        error = self.radToDeg(psisoll - psi)
        ddelta = self.trajectoryControler(error)
        deltac = deltac + self.degToRad(ddelta)
        dx = self.carModelOneLane(x, t0, control=False, constInput=True, uc=[vc, deltac])
        return dx

    def tryToOvertake(self, lastKnownFrontCar, xBackCar, t0):
        # first simulate the front car alone up to actual time t0 (try later if this can be safed persistently)
        n = xBackCar.size
        if t0 > 0:
            t0Range = np.arange(0, t0, 0.1)
            xFrontCar = self.simulateModel(lastKnownFrontCar.state, t0Range, model='complex', constInput=True,
                                           uc=[lastKnownFrontCar.v, lastKnownFrontCar.delta])
        else:
            xFrontCar = lastKnownFrontCar.state
        # then simulate both cars paralel for the time horizon of a lane change
        T = self.trajectory.specifics.T  # Time from trajectory
        TRange = np.arange(0, T, 0.1)
        xAugmented = np.concatenate((xBackCar, xFrontCar), axis=None)
        xBothCars = self.simulateModel(xAugmented, TRange, model='parallel', control=True,
                                       uc=[lastKnownFrontCar.v, lastKnownFrontCar.delta])
        xPosBackCar = xBothCars[-1, 0]
        xPosFrontCar = xBothCars[-1, n]
        return xPosBackCar - self.param.lh > xPosFrontCar


class TrajectorySpecifics:
    def __init__(self, Vsoll, W, name, Vstart=None, Psi=None, R=None):
        self.Vsoll = Vsoll
        self.name = name
        self.W = W
        if not Vstart:
            self.Vstart = Vsoll
        else:
            self.Vstart = Vstart
        self.Psi = Psi
        self.R = R
        self.T, self.D = self.VWtoDT()

    def VWtoDT(self):
        Vmiddle = abs(self.Vsoll + self.Vstart) / 2
        Vmax = 2.2
        if self.name == "parabolic":
            D = 4 * self.W
        else:
            D = 3*self.W
#            if Vmiddle <= Vmax / 4:
#                D = self.W
#            elif Vmax / 4 < Vmiddle <= Vmax:  # linear scaling of D between W and 3W
#                Vm = Vmiddle / Vmax
#                D = self.W * (8 / 3 * Vm + 1 / 3)
#            else:
#                D = 3 * self.W
        T = D / Vmiddle
        return abs(T), abs(D)

    def setVsoll(self, Vsoll):
        self.Vsoll = Vsoll
        self.T, self.D = self.VWtoDT()

    def setVstart(self, Vstart):
        self.Vstart = Vstart
        self.T, self.D = self.VWtoDT()

    def setW(self, W):
        self.W = W
        self.T, self.D = self.VWtoDT()

    def setVW(self, Vsoll=None, Vstart=None, W=None):
        if Vsoll:
            self.Vsoll = Vsoll
        if Vstart:
            self.Vstart = Vstart
        else:
            self.Vstart = Vsoll
        if W:
            self.W = W

        self.T, self.D = self.VWtoDT()

    def setPsi(self, Psi):
        self.Psi = Psi

    def setR(self, R):
        self.R = R

    def setAll(self, Vsoll=None, W=None, name=None, Vstart=None, Psi=None, R=None):
        if Vsoll:
            self.Vsoll = Vsoll
        if Vstart:
            self.Vstart = Vstart
        else:
            self.Vstart = Vsoll
        if W:
            self.W = W
        if name:
            self.name = name
        if Psi:
            self.Psi = Psi
        if R:
            self.R = R
        self.T, self.D = self.VWtoDT()


class trajectory:
    def __init__(self, Vsoll, W, name="cubicS", Vstart=None, Psi=None, R=None):
        # Vsoll to T
        self.specifics = TrajectorySpecifics(Vsoll, W, name=name, Vstart=Vstart, Psi=Psi, R=R)
        self.coeff = self.calcCoeff()

    def updateW(self, W):
        self.specifics.setW(W)
        self.coeff = self.calcCoeff()

    def updateVsoll(self, Vsoll):
        self.specifics.setVsoll(Vsoll)
        self.coeff = self.calcCoeff()

    def setSpecifics(self, Vsoll=None, W=None, name=None, Vstart=None, Psi=None, R=None):
        self.specifics.setAll(Vsoll=Vsoll, W=W, name=name, Vstart=Vstart, Psi=Psi, R=R)
        self.coeff = self.calcCoeff()

    def calcCoeff(self, specify=None, name=None):
        if not specify:
            specify = self.specifics
        if not name:
            name = self.specifics.name
        if not specify.Psi or not specify.R:
            Wp = specify.W
            m = 0
#        else:
#            Wp = specify.R - np.cos(specify.Psi) * (specify.R - specify.W)
#            m = np.tan(specify.Psi)

        if name == "parabolic":

            A = np.array([(specify.T ** 2) / 9, -1, -specify.T / 3, -(specify.T ** 2) / 9, 0, 0, 0,
                          2 * specify.T / 3, 0, -1, -2 * specify.T / 3, 0, 0, 0,
                          0, 1, specify.T / 2, (specify.T ** 2) / 4, 0, 0, 0,
                          0, 0, 1, specify.T, 0, 0, 0,
                          0, 1, 2 * specify.T / 3, 4 * (specify.T ** 2) / 9, -1, -2 * specify.T / 3,
                          -4 * (specify.T ** 2) / 9,
                          0, 0, 1, 4 * specify.T / 3, 0, -1, -4 * specify.T / 3,
                          0, 0, 0, 0, 1, specify.T, specify.T ** 2]).reshape(7, 7)  # 0, 0, 0, 0, 0, 1, 2*T
            b = np.zeros(A.shape[0])
            b[2] = specify.W
            coeff = np.linalg.solve(A, b)
        elif name == "quad":
            A = np.array([1, 0, 0,
                          1, specify.T, specify.T ** 2,
                          0, 1, 2 * specify.T]).reshape(3, 3)  # 0, 0, 0, 0, 0, 1, 2*T

            b = np.array([0, Wp, m])
            coeff = (np.linalg.solve(A, b))
        elif name == "quadS":
            # if self.specifics.Psi:
            A = np.array([(specify.T ** 2) / 4, -1, -specify.T / 2, -(specify.T ** 2) / 4,
                          2 * specify.T / 2, 0, -1, -2 * specify.T / 2,
                          0, 1, specify.T, (specify.T ** 2),
                          0, 0, 1, 2 * specify.T]).reshape(4, 4)  # 0, 0, 0, 0, 0, 1, 2*T

            b = np.zeros(A.shape[0])
            b[2] = Wp
            b[3] = m
            coeff = (np.linalg.solve(A, b))
        elif name == "sShape":
            A = np.array([specify.T ** 3, specify.T ** 2,
                          3 * specify.T ** 2, 2 * specify.T]).reshape(2, 2)
            # a3 = -2*specify.W/specify.T**3
            # a2 = 3*specify.W/specify.T**2
            b = np.array([Wp, m])
            coeff = np.linalg.solve(A, b)
            # coeff = np.array([a3,a2])

        elif name == "cubicS":
            T1 = specify.T / 2
            A = np.array([T1 ** 2, T1 ** 3, -1, -T1, -T1 ** 2, -T1 ** 3,
                          T1 * 2, 3 * T1 ** 2, 0, -1, -2 * T1, -3 * T1 ** 2,
                          2, 6 * T1, 0, 0, -2, -6 * T1,
                          0, 0, 1, specify.T, specify.T ** 2, specify.T ** 3,
                          0, 0, 0, 1, 2 * specify.T, 3 * specify.T ** 2,
                          0, 0, 0, 0, 2, 6 * specify.T]).reshape(6, 6)

            b = np.zeros(A.shape[0])
            b[3] = Wp
            b[4] = m
            coeff = np.linalg.solve(A, b)
        elif name == 'nineS' or name == 'nineP':
            A = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 6, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 24, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          0, 0, 2, 6, 12, 20, 30, 42, 56, 72,
                          0, 0, 0, 6, 24, 60, 120, 210, 336, 504,
                          0, 0, 0, 0, 24, 120, 360, 840, 1680, 3024]).reshape(10, 10)
            b = np.zeros(A.shape[0])
            b[0] = 0  # 0 for y, 0 for x
            b[1] = 0  # 0.01 for y  Vstart for x
            b[5] = Wp  # W for y D for x D should depend on V
            b[6] = 0  # 0 for y Vsoll for x
            coeff = np.linalg.solve(A, b)
            if name == 'nineP': # nineP is a parametrized path therefore x and y a polynomes
                b[0] = 0  # 0 for y, 0 for x
                b[1] = specify.Vstart  # 0.01 for y  Vstart for x
                b[5] = specify.D  # W for y D for x D should depend on V
                b[6] = specify.Vsoll  # 0 for y Vsoll for x
                coeffx = np.linalg.solve(A, b)
                coeff = np.stack((coeff,coeffx))

        else:
            raise NameError('given trajecorty does not match any implemented one')
        return coeff

    def generateSpline(self, t, derivative='zero'):
        if self.specifics.name == "parabolic":
            return self._parabolaSpline(t, derivative=derivative)
        elif self.specifics.name == "quadS":
            return self._quadsSpline(t, derivative=derivative)
        elif self.specifics.name == "sShape":
            return self._sShape(t, derivative=derivative)
        elif self.specifics.name == "cubicS":
            return self._cubicsSpline(t, derivative=derivative)
        elif self.specifics.name == 'nineS':
            return self._nineSpline(t,derivative=derivative)
        else:
            raise NameError('given trajecorty does not match any implemented one')

    def _quadsSpline(self, t, specify=None, derivative='zero'):
        if not specify:
            specify = self.specifics

        if derivative == 'zero':
            if 0 <= t < specify.T / 2:
                y = self.coeff[0] * t ** 2
            elif specify.T / 2 <= t < specify.T:
                y = self.coeff[1] + self.coeff[2] * t + self.coeff[3] * t ** 2
            else:
                y = self.coeff[1] + self.coeff[2] * specify.T + self.coeff[3] * specify.T ** 2
        elif derivative == 'first':
            if 0 <= t < specify.T / 2:
                y = 2 * self.coeff[0] * t
            elif specify.T / 2 <= t < specify.T:
                y = self.coeff[2] + 2 * self.coeff[3] * t
            else:
                y = 0
        elif derivative == 'second':
            if 0 <= t < specify.T / 2:
                y = 2 * self.coeff[0]
            elif specify.T / 2 <= t < specify.T:
                y = 2 * self.coeff[3]
            else:
                y = 0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y

    def _cubicsSpline(self, t, specify=None, derivative='zero'):
        if not specify:
            specify = self.specifics

        if derivative == 'zero':
            if 0 <= t < specify.T / 2:
                y = self.coeff[0] * t ** 2 + self.coeff[1] * t ** 3
            elif specify.T / 2 <= t <= specify.T:
                y = self.coeff[2] + self.coeff[3] * t + self.coeff[4] * t ** 2 + self.coeff[5] * t ** 3
            else:
                y = self.coeff[2] + self.coeff[3] * specify.T + self.coeff[4] * specify.T ** 2 + self.coeff[
                    5] * specify.T ** 3
        elif derivative == 'first':

            if 0 <= t < specify.T / 2:
                y = 2 * self.coeff[0] * t + 3 * self.coeff[1] * t ** 2
            elif specify.T / 2 <= t <= specify.T:
                y = self.coeff[3] + 2 * self.coeff[4] * t + 3 * self.coeff[5] * t ** 2
            else:
                y = 0
        elif derivative == 'second':
            if 0 <= t < specify.T / 2:
                y = 2 * self.coeff[0] + 6 * self.coeff[1] * t
            elif specify.T / 2 <= t <= specify.T:
                y = 2 * self.coeff[4] + 6 * self.coeff[5] * t
            else:
                y = 0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y

    def _sShape(self, t, specify=None, derivative="zero"):
        if not specify:
            specify = self.specifics

        if derivative == 'zero':
            if 0 <= t <= specify.T:
                y = self.coeff[0] * t ** 3 + self.coeff[1] * t ** 2
            else:
                y = self.coeff[0] * specify.T ** 3 + self.coeff[1] * specify.T ** 2
        elif derivative == 'first':
            if 0 <= t <= specify.T:
                y = 3 * self.coeff[0] * t ** 2 + 2 * self.coeff[1] * t
            else:
                y = 0
        elif derivative == 'second':
            if 0 <= t <= specify.T:
                y = 6 * self.coeff[0] * t + 2 * self.coeff[1]
            else:
                y = 0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y

    def _parabolaSpline(self, t, specify=None, derivative='zero'):
        if not specify:
            specify = self.specifics

        if derivative == 'zero':
            if 0 <= t < specify.T / 3:
                y = self.coeff[0] * t ** 2
            elif specify.T / 3 <= t < 2 * specify.T / 3:
                y = self.coeff[1] + self.coeff[2] * t + self.coeff[3] * t ** 2
            elif 2 * specify.T / 3 <= t <= specify.T:
                y = self.coeff[4] + self.coeff[5] * t + self.coeff[6] * t ** 2
            else:
                y = 0
        elif derivative == 'first':
            if 0 <= t < specify.T / 3:
                y = 2 * self.coeff[0] * t
            elif specify.T / 3 <= t < 2 * specify.T / 3:
                y = self.coeff[2] + 2 * self.coeff[3] * t
            elif 2 * specify.T / 3 <= t <= specify.T:
                y = self.coeff[5] + 2 * self.coeff[6] * t
            else:
                y = 0
        elif derivative == 'second':
            if 0 <= t < specify.T / 3:
                y = 2 * self.coeff[0]
            elif specify.T / 3 <= t < 2 * specify.T / 3:
                y = 2 * self.coeff[3]
            elif 2 * specify.T / 3 <= t <= specify.T:
                y = 2 * self.coeff[6]
            else:
                y = 0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y

    def _nineSpline(self, t, derivative='zero'):
        return self._ninePolynom(t,self.specifics.T,derivative=derivative)

    def _ninePolynom(self, t, normalizer, derivative='zero',coeffaxis=None):
        if coeffaxis is None:
            coeff= self.coeff
        else:
            coeff = self.coeff[coeffaxis,:]
        v = t / normalizer
        vpower = np.array([v ** i for i in np.arange(0, 10)])
        if derivative == 'zero':
            if 0 <= v <= 1:
                #s = self._polyHorner(v,coeff)
                s = np.sum(coeff * vpower)
                #print(s==s1)
            else:
                s = np.sum(coeff)
        elif derivative == 'first':
            if 0 <= v <= 1:

                coeff = coeff[1:] * np.arange(1, 10)
                #s = self._polyHorner(v,coeff)
                s = np.sum(coeff * vpower[:-1])
                #print(s==s1)
            else:
                s=0
        elif derivative == 'second':
            if 0 <= v <= 1:
                coeff = coeff[1:] * np.arange(1, 10)
                coeff = coeff[1:] * np.arange(1, 9)
                s = np.sum(coeff * vpower[:-2])
                #s = self._polyHorner(v,coeff)
                #print(s==s1)
            else:
                s=0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return s

    def _polyHorner(self, x, coeff):
        result = coeff[-1]
        for i in range(-2, -len(coeff) - 1, -1):
            result = result * x + coeff[i]
        return result

    def generateXtrajectory(self, t, derivative='zero'):

        if not self.specifics.Vstart:
            self.specifics.setVstart(self.specifics.Vsoll)
        if derivative == 'zero':
            if 0 <= t <= self.specifics.T:
                x = 0.5 * (
                            self.specifics.Vsoll - self.specifics.Vstart) * t ** 2 + self.specifics.Vstart * t  # assume sim starts with x=0
            elif t < 0:
                x = self.specifics.Vstart * np.abs(t)
            else:
                x = self.specifics.Vsoll * t
        elif derivative == 'first':
            if 0 <= t <= self.specifics.T:
                x = (self.specifics.Vsoll - self.specifics.Vstart) * t + self.specifics.Vstart
            elif t < 0:
                x = self.specifics.Vstart
            else:
                x = self.specifics.Vsoll
        elif derivative == 'second':
            if 0 <= t <= self.specifics.T:
                x = self.specifics.Vsoll - self.specifics.Vstart
            else:
                x = 0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return x

    def generateParametrizedPath(self,t,x,derivative='zero'):
        x_t = self._ninePolynom(t,self.specifics.T, derivative=derivative,coeffaxis=0)
        if derivative == 'zero':
            y_x = self._ninePolynom(x_t,self.specifics.D, derivative=derivative,coeffaxis=1)
        else:
            y_x = self._ninePolynom(x, self.specifics.D, derivative=derivative,coeffaxis=1)

        return x_t,y_x