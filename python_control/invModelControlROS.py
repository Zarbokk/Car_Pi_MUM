# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:00:27 2019

@author: Felix
"""

import numpy as np
# import calcSpline
import scipy.integrate

class parameters:
    def __init__(self,m, Iz, lv, lh, EG):
        self.m = m # Masse in kg
        self.Iz = Iz # Trägheitsmoment in kg*m**2
        self.lv = lv # abstand von schwerpunkt zu vorderreifen in m
        self.lh = lh # abstand von schwerpunkt zu hinterreifen in m
        self.L = lv+lh # gesamtlänge in m
        self.EG = EG # eigenlenkgradient aus parameterident
        self.ks = (lv-lh)/EG/self.L*m  # Schräglaaufsteiigkeit vorne und hinten

class virtualCarContainer:
    def __init__(self,x,v,delta):
        self.state = x
        self.v = v
        self.delta = delta


class invModelControl:
    def __init__(self,Vsoll,W,trajectoryType="cubicS"):
        m = 2.26113  # Masse in kg
        Iz = 0.0274  # Trägheitsmoment in kg*m**2
        lv = 0.16812  # abstand von schwerpunkt zu vorderreifen
        lh = 0.11687  # abstand von schwerpunkt zu hinterreifen
        EG = 0.0547  # eigenlenkgradient aus parameterident
        self.param=parameters(m, Iz, lv, lh, EG)
        self.Vsoll = Vsoll
        self.trajectory = trajectory((Vsoll,W),name=trajectoryType)
        self.overtake = False
        self.T0 = None
        self.accSoll = 0.5
        self.lastKnownFrontCar = virtualCarContainer(np.array([0, W, 0, 0, 0]),Vsoll,0)

    ################
    # first all functions for the changelane maneuver
    ################

    def carInput(self,t,phase='changeLane'):
        if phase == 'changeLane':
            p,dp,ddp = self.trajectoryGen(t)
            v,delta,psi = self.invModel(p,dp,ddp)
            if delta > self.degToRad(29):
                delta = self.degToRad(29)
            elif delta < self.degToRad(-29):
                delta = self.degToRad(-29)
        elif phase == 'straightLine':
            v = self.Vsoll+self.accSoll*t
            delta = 0
            psi = 0
        return v, delta, psi

    def trajectoryControler(self,error,p=10):
        maxerror = 30
        maxsteering = 10
        errorConv = error/maxerror*maxsteering
        return p*errorConv

    def invModel(self,p, dp, ddp):
        # dxsoll**2+dysoll**2 unequal 0
        # p is the desired trajectory dp ddp are derivative w.r.t. time
        dxsoll = dp[0]
        dysoll = dp[1]
        ddxsoll = ddp[0]
        ddysoll = ddp[1]
        psi = np.arctan(dysoll/dxsoll)
        dpsi = (dxsoll*ddysoll-dysoll*ddxsoll)/(dxsoll**2+dysoll**2)
        v = dxsoll*np.cos(psi)+dysoll*np.sin(psi)
        delta = np.arctan(self.param.L*dpsi/v)
        return v, delta, psi

    def trajectoryGen(self,t):
        xsoll = self.Vsoll*t
        dxsoll = self.Vsoll
        ddxsoll = 0
        ysoll = self.trajectory.generateSpline(t)
        dysoll = self.trajectory.generateSpline(t, derivative='first')
        ddysoll = self.trajectory.generateSpline(t, derivative='second')

        p = np.array([xsoll, ysoll]).reshape(2)
        dp = np.array([dxsoll, dysoll]).reshape(2)
        ddp = np.array([ddxsoll, ddysoll]).reshape(2)
        return p, dp, ddp

    def radToDeg(self,x):
        return x/np.pi*180

    def degToRad(self,x):
        return x*np.pi/180

    ################
    # all functions for simulating while overtaking
    ################

    def simulateModel(self,y0,trange,model='complex', **kwargs):
        if model == 'complex':
            if y0.size < 5:
               y0=np.concatenate((y0,np.zeros((5-y0.size))),axis=None)
            try:
                ode = lambda x,t : self.carModelOneLane(x,t,**kwargs)
            except TypeError as error:
               print(error)
            y = scipy.integrate.odeint(ode, y0, trange)
        elif model == 'parallel':
            if y0.size < 10:
                y0=np.concatenate((y0,np.zeros((10-y0.size,))),axis=None)
            try:
                ode = lambda x,t : self.carModelParallel(x,t,**kwargs)
            except TypeError as error:
                print(error)

            y = scipy.integrate.odeint(ode, y0, trange)
        elif model == 'full':
            if y0.size < 5:
                y0=np.concatenate((y0,np.zeros((5-y0.size,))),axis=None)
            try:
                ode = lambda x,t : self.carModelFull(x,t)
            except TypeError as error:
                print(error)

            y = scipy.integrate.odeint(ode, y0, trange)
        elif model == 'simple':
            y0=y0[:3]
            try:
                ode = lambda x,t : self.carModelOneLaneSimple(x,t,**kwargs)
            except TypeError as error:
                print(error)
            y = scipy.integrate.odeint(ode, y0, trange)
        else:
             raise NameError('given model type does not match any implemented one')
        return y

    def carModelOneLaneSimple(self,x,t0,control=True, constInput=False,uc=[1,0]):
#        if 'control' in kwargs.keys():
#            control = kwargs['control']
#        else:
#            control = False
        Xpos = x[0]
        Ypos = x[1]
        psi = x[2]
        if  constInput:
            v = uc[0]
            delta = uc[1]
        else:
            v, delta,psisoll = self.carInput(t0)
            if control:
                error = self.radToDeg(psisoll-psi)
                ddelta = self.trajectoryControler(error)
                delta = delta+self.degToRad(ddelta)

        dx = np.array([np.cos(psi), np.sin(psi), np.tan(delta)/self.param.L])*v
        return dx

    def carModelOneLane(self,x, t0,control=True, constInput=False,uc=[1,0]):
#        if 'constInput' in kwargs.keys():
#            constInput = kwargs['constInput']
#            if 'uc' in
#        else:
#            constInput = False
        #%% input and constants
        vTol = 1.2*10**(-3) #toleranz for small velocitys to switch to simple model
        #%% states
        xpos = x[0]
        ypos = x[1]
        betha = x[2]
        psi = x[3]
        phi = x[4] # phi is dummy for dpsi
        if  constInput:
            v = uc[0]
            delta = uc[1]
        else:
            v, delta,psisoll = self.carInput(t0)
            if control:
                error = self.radToDeg(psisoll-psi)
                ddelta = self.trajectoryControler(error)
                delta = delta+self.degToRad(ddelta)

        if v == 0:
            v=vTol
        v = np.sign(v)*np.max([np.abs(v),vTol])
        if(np.abs(v)>=vTol):
            #%% forces (possible to incorporate  more)
            alpha_v = delta-betha-self.param.lv*phi/v
            alpha_h = self.param.lh*phi/v-betha
            Fv = self.param.ks*self.radToDeg(alpha_v) # ks is in N/deg
            Fh = self.param.ks*self.radToDeg(alpha_h)
            Fx = -Fv*np.sin(delta)
            Fy = Fh+np.cos(delta)*Fv
            #%% equations
            dpsi = phi
            dbetha = (Fy*np.cos(betha)**2-np.cos(betha)*np.sin(betha)*Fx)/(self.param.m*v)-dpsi
            dphi = (np.cos(delta)*Fv*self.param.lv-Fh*self.param.lh)/self.param.Iz
            dxpos = np.tan(betha)*np.sin(psi)*v+np.cos(psi)*v
            dypos = -np.tan(betha)*np.cos(psi)*v+np.sin(psi)*v
        else:
            smallx=np.array([xpos, ypos, psi])
            dx = self.carModelOneLaneSimple(smallx,t0)
            dxpos = dx[0]# np.cos(psi)*v
            dypos = dx[1]#np.sin(psi)*v
            dbetha = 0
            dpsi = dx[2]#0
            dphi = 0
        return [dxpos, dypos, dbetha, dpsi, dphi]

    def carModelParallel(self,xa,t0,control=True,uc=[1,0]):
        n=xa.size//2
        dx1 = self.carModelOneLane(xa[:n],t0,control=control)
        dx2 = self.carModelOneLane(xa[n:],t0,constInput=True,uc=uc)
        return np.concatenate((dx1,dx2),axis=None)

    def carModelFull(self,x,t0):
        psi = x[3]
        #%% externaly call carInput depending on what the internal simulation says
        if not self.T0: # T0 is only initialized once as None
            self.overtake = self.tryToOvertake(self.lastKnownFrontCar,x,t0)
            if self.overtake:
                self.T0 = t0 # save overtake starting time
                print(t0)
        if self.overtake == True:
            vc,deltac,psisoll =  self.carInput(t0-self.T0) # get carInout due to trajectory change relative to start overtake T0
        else:
            vc, deltac, psisoll = self.carInput(t0,phase='straightLine')
        error = self.radToDeg(psisoll-psi)
        ddelta = self.trajectoryControler(error)
        deltac = deltac+self.degToRad(ddelta)
        dx = self.carModelOneLane(x, t0,control=False,constInput=True,uc=[vc,deltac])
        return dx

    def tryToOvertake(self,lastKnownFrontCar,xBackCar,t0):
        # first simulate the front car alone up to actual time t0 (try later if this can be safed persistently)
        n=xBackCar.size
        if t0 > 0:
            t0Range = np.arange(0,t0,0.1)
            xFrontCar = self.simulateModel(lastKnownFrontCar.state, t0Range, model='complex', constInput=True, uc=[lastKnownFrontCar.v,lastKnownFrontCar.delta])
        else:
            xFrontCar = lastKnownFrontCar.state
        # then simulate both cars paralel for the time horizon of a lane change
        T = self.trajectory.specifics[0] # Time from trajectory
        TRange = np.arange(0,T,0.1)
        xAugmented = np.concatenate((xBackCar, xFrontCar),axis=None)
        xBothCars = self.simulateModel(xAugmented,TRange,model='parallel',control=True, uc=[lastKnownFrontCar.v,lastKnownFrontCar.delta])
        xPosBackCar = xBothCars[-1,0]
        xPosFrontCar = xBothCars[-1,n]
        return xPosBackCar-self.param.lh > xPosFrontCar


class trajectory:
    def __init__(self,specify ,name="cubicS",):
        # Vsoll to T
        Vsoll = specify[0]

        if name == "parabolic":
            T = 4*specify[1]/Vsoll
        elif name == "quadS":
            T = 2*specify[1]/Vsoll
        elif name == "sShape":
            T = 2*specify[1]/Vsoll
        elif name == "cubicS":
            T = 2*specify[1]/Vsoll
        else:
            raise NameError('given trajecorty does not match any implemented one')
        self.specifics = (T,specify[1])
        self.name = name
        self.coeff = self.calcCoeff()

    def calcCoeff(self,specify=None, name = None):
        if not specify:
            specify = self.specifics
        if not name:
            name = self.name
        if name == "parabolic":
            T=specify[0]
            W=specify[1]
            A = np.array([(T**2)/9, -1, -T/3, -(T**2)/9, 0, 0, 0,
                            2*T/3, 0, -1, -2*T/3, 0, 0, 0,
                            0, 1, T/2, (T**2)/4, 0, 0, 0,
                            0, 0, 1, T, 0, 0, 0,
                            0, 1, 2*T/3, 4*(T**2)/9, -1, -2*T/3, -4*(T**2)/9,
                            0, 0, 1, 4*T/3, 0, -1, -4*T/3,
                            0, 0, 0, 0, 1, T, T**2]).reshape(7, 7) # 0, 0, 0, 0, 0, 1, 2*T
            b = np.zeros(A.shape[0])
            b[2] = W
            coeff = np.linalg.solve(A, b)
        elif name == "quadS":
            T=specify[0]
            W=specify[1]
            A = np.array([(T**2)/4, -1, -T/2, -(T**2)/4,
                            2*T/2, 0, -1, -2*T/2,
                            0, 1, T, (T**2),
                            0, 0, 1, 2*T,]).reshape(4, 4) # 0, 0, 0, 0, 0, 1, 2*T

            b = np.zeros(A.shape[0])
            b[2] = W
            coeff = (np.linalg.solve(A, b))
        elif name == "sShape":
            T = specify[0]
            W = specify[1]
            a3 = -2*W/T**3
            a2 = 3*W/T**2
            coeff = np.array([a3,a2])

        elif name == "cubicS":
            T = specify[0]
            T1 = T/2
            W = specify[1]
            A = np.array([T1**2, T1**3, -1, -T1,-T1**2,-T1**3,
                            T1*2, 3*T1**2, 0, -1,-2*T1,-3*T1**2,
                            2, 6*T1, 0, 0,-2,-6*T1,
                            0, 0, 1, T, T**2, T**3,
                            0, 0, 0, 1, 2*T, 3*T**2,
                            0, 0, 0, 0, 2, 6*T]).reshape(6, 6)

            b = np.zeros(A.shape[0])
            b[3] = W
            coeff = np.linalg.solve(A, b)
        else:
            raise NameError('given trajecorty does not match any implemented one')
        return coeff

    def generateSpline(self,t,derivative='zero'):
        if self.name == "parabolic":
            return self._parabolaSpline(t, derivative=derivative)
        elif self.name == "quadS":
            return self._quadsSpline(t, derivative=derivative)
        elif self.name == "sShape":
            return self._sShape(t,derivative=derivative)
        elif self.name == "cubicS":
            return self._cubicsSpline(t,derivative=derivative)
        else:
            raise NameError('given trajecorty does not match any implemented one')

    def _quadsSpline(self, t, specify = None, derivative='zero'):
        if not specify:
            specify = self.specifics

        T = specify[0]
        if derivative == 'zero':
            if t>=0 and t<T/2:
                y = self.coeff[0]*t**2
            elif t>=T/2 and t<T:
                y = self.coeff[1]+self.coeff[2]*t+self.coeff[3]*t**2
            else:
                y= self.coeff[1]+self.coeff[2]*T+self.coeff[3]*T**2
        elif derivative == 'first':
            if t>=0 and t<T/2:
                y = 2*self.coeff[0]*t
            elif t>=T/2 and t<T:
                y = self.coeff[2]+2*self.coeff[3]*t
            else:
                y=0
        elif derivative == 'second':
            if t>=0 and t<T/2:
                y = 2*self.coeff[0]
            elif t>=T/2 and t<T:
                y = 2*self.coeff[3]
            else:
                y=0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y


    def _cubicsSpline(self, t, specify = None, derivative='zero'):
        if not specify:
            specify = self.specifics

        T = specify[0]
        if derivative == 'zero':
            if t>=0 and t<T/2:
                y = self.coeff[0]*t**2+self.coeff[1]*t**3
            elif t>=T/2 and t<=T:
                y = self.coeff[2]+self.coeff[3]*t+self.coeff[4]*t**2+self.coeff[5]*t**3
            else:
                y = self.coeff[2]+self.coeff[3]*T+self.coeff[4]*T**2+self.coeff[5]*T**3
        elif derivative == 'first':

            if t>=0 and t<T/2:
                y = 2*self.coeff[0]*t+3*self.coeff[1]*t**2
            elif t>=T/2 and t<=T:
                y = self.coeff[3]+2*self.coeff[4]*t+3*self.coeff[5]*t**2
            else:
                y=0
        elif derivative == 'second':
            if t>=0 and t<T/2:
                y = 2*self.coeff[0]+6*self.coeff[1]*t
            elif t>=T/2 and t<=T:
                y = 2*self.coeff[4]+6*self.coeff[5]*t
            else:
                y=0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y

    def _sShape(self, t,specify = None, derivative="zero"):
        if not specify:
            specify = self.specifics
        T = specify[0]

        if derivative == 'zero':
            if t>=0 and t<=T:
                y = self.coeff[0]*t**3+self.coeff[1]*t**2
            else:
                y = self.coeff[0]*T**3+self.coeff[1]*T**2
        elif derivative == 'first':
            if t>=0 and t<=T:
                y = 3*self.coeff[0]*t**2+2*self.coeff[1]*t
            else:
                y=0
        elif derivative == 'second':
            if t>=0 and t<=T:
                y = 6*self.coeff[0]*t+2*self.coeff[1]
            else:
                y = 0
        else:
           raise NameError('requested derivative does not match any implemented one')
        return y

    def _parabolaSpline(self, t, specify=None, derivative='zero'):
        if not specify:
            specify = self.specifics

        T = specify[0]
        if derivative == 'zero':
            if t>=0 and t<T/3:
                y = self.coeff[0]*t**2
            elif t>=T/3 and t<2*T/3:
                y = self.coeff[1]+self.coeff[2]*t+self.coeff[3]*t**2
            elif t>=2*T/3 and t<=T:
                y = self.coeff[4]+self.coeff[5]*t+self.coeff[6]*t**2
            else:
                y=0
        elif derivative == 'first':
            if t>=0 and t<T/3:
                y = 2*self.coeff[0]*t
            elif t>=T/3 and t<2*T/3:
                y = self.coeff[2]+2*self.coeff[3]*t
            elif t>=2*T/3 and t<=T:
                y = self.coeff[5]+2*self.coeff[6]*t
            else:
                y=0
        elif derivative == 'second':
            if t>=0 and t<T/3:
                y = 2*self.coeff[0]
            elif t>=T/3 and t<2*T/3:
                y = 2*self.coeff[3]
            elif t>=2*T/3 and t<=T:
                y = 2*self.coeff[6]
            else:
                y=0
        else:
            raise NameError('requested derivative does not match any implemented one')
        return y







