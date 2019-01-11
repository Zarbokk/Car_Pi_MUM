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



class invModelControl:
    def __init__(self,Vsoll,W,trajectoryType="qubic"):
        m = 2.26113  # Masse in kg
        Iz = 0.0274  # Trägheitsmoment in kg*m**2
        lv = 0.16812  # abstand von schwerpunkt zu vorderreifen
        lh = 0.11687  # abstand von schwerpunkt zu hinterreifen
        EG = 0.0547  # eigenlenkgradient aus parameterident
        self.param=parameters(m, Iz, lv, lh, EG)
        self.Vsoll = Vsoll
        self.trajectory = trajectory((Vsoll,W),name=trajectoryType)

    ################
    # first all functions for the openloop changelane maneuver
    ################

    def carInput(self,t):
        p,dp,ddp = self.trajectoryGen(t)
        v,delta = self.invModel(p,dp,ddp)
        if delta > self.degToRad(29):
            delta = self.degToRad(29)
        elif delta < self.degToRad(-29):
            delta = self.degToRad(-29)
        return v, delta

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
        return v, delta

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

    def simulateModel(self,y0,trange,model='complex'):
        if model == 'complex':
            if y0.size < 5:
                y0=np.array([y0,np.zeros((5-y0.size,))])
            y = scipy.integrate.odeint(self.carModelOneLane, y0, trange)
        elif model == 'parralel':
            if y0.size < 10:
                y0=np.array([y0,np.zeros((10-y0.size,))])
            y = scipy.integrate.odeint(self.carModelParallel, y0, trange)
        else:
            y0=y0[:3]
            y = scipy.integrate.odeint(self.carModelOneLaneSimple, y0, trange)
        return y

    def carModelOneLaneSimple(self,x,t0):
        Xpos = x[0]
        Ypos = x[1]
        psi = x[2]
        v, delta = self.carInput(t0)
        dx = np.array([np.cos(psi), np.sin(psi), np.tan(delta)/self.param.L])*v
        return dx

    def carModelOneLane(self,x, t0,constInput=False,vc=0,deltac=0):
        #%% input and constants
        vTol = 1.2*10**(-3) #toleranz for small velocitys to switch to simple model
        if  constInput:
            v = vc
            delta = deltac
        else:
            v, delta = self.carInput(t0)

        #%% states
        xpos = x[0]
        ypos = x[1]
        betha = x[2]
        psi = x[3]
        phi = x[4] # phi is dummy for dpsi
        if(np.abs(v)>vTol):
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

    def carModelParallel(self,xa,t0):
        n=len(xa)
        x1=xa[:n/2] # states of first car
        x2=xa[n/2:] # states of second car
        dx1 = self.carModelOneLane(x1,t0)
        dx2 = self.carModelOneLane(x2,t0,constInput = True,vc = 5,deltac = 0)
        return np.vstack((dx1,dx2))

class trajectory:
    def __init__(self,specify ,name="qubic",):
        # Vsoll to T
        Vsoll = specify[0]
        if name == "qubic":
            T = 4*specify[1]/Vsoll
        elif name == "qubicS":
            T = 2*specify[1]/Vsoll
        elif name == "sShape":
            T = 2*specify[1]/Vsoll
        self.specifics = (T,specify[1])
        self.name = name
        self.coeff = self.calcCoeff()

    def calcCoeff(self,specify=None, name = None):
        if not specify:
            specify = self.specifics
        if not name:
            name = self.name
        if name == "qubic":
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
        elif name == "qubicS":
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
        return coeff

    def generateSpline(self,t,derivative='zero'):
        if self.name == "qubic":
            return self._parabolaSpline(t, derivative=derivative)
        elif self.name == "qubicS":
            return self._qubicsSpline(t, derivative=derivative)
        elif self.name == "sShape":
            return self._sSpline(t,derivative=derivative)

    def _qubicsSpline(self, t, specify = None, derivative='zero'):
        if not specify:
            specify = self.specifics

        T = specify[0]
        if derivative == 'first':
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
            if t>=0 and t<T/2:
                y = self.coeff[0]*t**2
            elif t>=T/2 and t<T:
                y = self.coeff[1]+self.coeff[2]*t+self.coeff[3]*t**2
            else:
                y= self.coeff[1]+self.coeff[2]*T+self.coeff[3]*T**2
        return y


    def _sSpline(self, t, specify = None, derivative='zero'):
        if not specify:
            specify = self.specifics

        T = specify[0]
        if derivative == 'fist':
            if t>=0 and t<T:
                y = 3*self.coeff[0]*t**2+2*self.coeff[1]*t
            else:
                y=0
        elif derivative == 'second':
            if t>=0 and t<T:
                y = 6*self.coeff[0]*t+2*self.coeff[1]
            else:
                y=0
        else:
            if t>=0 and t<T:
                y = self.coeff[0]*t**3+self.coeff[1]*t**2
            else:
                y = self.coeff[0]*T**3+self.coeff[1]*T**2
        return y
    def _parabolaSpline(self, t, specify=None, derivative='zero'):
        if not specify:
            specify = self.specifics

        T = specify[0]
        if derivative == 'first':
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
            if t>=0 and t<T/3:
                y = self.coeff[0]*t**2
            elif t>=T/3 and t<2*T/3:
                y = self.coeff[1]+self.coeff[2]*t+self.coeff[3]*t**2
            elif t>=2*T/3 and t<=T:
                y = self.coeff[4]+self.coeff[5]*t+self.coeff[6]*t**2
            else:
                y=0
        return y







