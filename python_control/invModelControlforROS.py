# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:28:49 2019

@author: Felix
"""
import numpy as np
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

class invModelControlROS:
    '''
    assumption : class wird neu initialisiert im ros mainloop
    '''
    def __init__(self, W, Vsoll):
        self.W = W # lanechange width
        self.T = 4*W/Vsoll # lanechange time
        self.Vsoll = Vsoll # lanechange start and end velocity expect the actual velocity
        # constant parameters
        m = 2.26113  # Masse in kg
        Iz = 0.0274  # Trägheitsmoment in kg*m**2
        lv = 0.16812  # abstand von schwerpunkt zu vorderreifen in m
        lh = 0.11687  # abstand von schwerpunkt zu hinterreifen in m
        EG = 0.0547  # eigenlenkgradient aus parameterident
        self.param = parameters(m, Iz, lv ,lh, EG)


    def _carModelParallel(self,xa,t0):
        '''
        two cars where x1 is doing a lane change and x2 is driving constant with vc deltac
        '''
        n=len(xa)
        x1=xa[:n/2] # states of first car
        x2=xa[n/2:] # states of second car
        dx1 = self._carModelOneLane(x1,t0)
        dx2 = self._carModelOneLane(x2, t0, constInput=True, vc=5, deltac=0)

    def simulateCars(self, x01, x02, trange):
        step = 0.01
        t = np.arange(0,trange,step)
        if not x02:
            y = scipy.integrate.odeint(self._carModelOneLane, x01, t)
            return y
        else:
            x0a = np.hstack((x01,x02))
            y = scipy.integrate.odeint(self._carModelParallel, x0a, t)
            return y[:,:len(x01)], y[:,len(x01):]

    def tryToOvertake(self, x01, x02):
        y1, y2 = self.simulateCars(x01, x02, self.T)
        if (y1[-1:0]-self.param.L)>=y2[-1:0]:
            return True
        else:
            return False

    def manageOvertake(self, x01, x02, t):
        overtake = self.tryToOvertake(x01, x02)
        if overtake:
            u = self.carInput(t)
            v = u[0]
            delta = u[1]
        else:
            v = self.Vsoll
            delta = 0
        return v, delta


    def _carModelOneLane(self,x, t0,constInput=False,vc=0,deltac=0):
            #%% input and constants
            if  constInput:
                v = vc
                delta = deltac
            else:
                u = self.carInput(t0)
                v = u[0]
                delta = u[1]
                # constant parameters
            m = self.param.m
            Iz = self.param.Iz
            lv = self.param.lv
            lh = self.param.lh
            L = self.param.L
            EG = self.param.EG
            ks = self.param.ks

            #%% states
            xpos = x[0]
            ypos = x[1]
            betha = x[2]
            psi = x[3]
            phi = x[4] # phi is dummy for dpsi

            if(delta!=0 and v != 0):
                #%% forces (possible to incorporate  more)
                alpha_v = delta-betha-lv*phi/v
                alpha_h = lh*phi/v-betha
                Fv = ks*self.radToDeg(alpha_v) # ks is in N/deg
                Fh = ks*self.radToDeg(alpha_h)
                Fx = -Fv*np.sin(delta)
                Fy = Fh+np.cos(delta)*Fv
                #%% equations
                dpsi = phi
                dbetha = (Fy*np.cos(betha)**2-np.cos(betha)*np.sin(betha)*Fx)/(m*v)-dpsi
                dphi = (np.cos(delta)*Fv*lv-Fh*lh)/Iz
                dxpos = np.tan(betha)*np.sin(psi)*v+np.cos(psi)*v
                dypos = -np.tan(betha)*np.cos(psi)*v+np.sin(psi)*v
            else:
                dx = self.carModelOneLaneSimple(x[:3],t0)
                dxpos = dx[0]
                dypos = dx[1]
                dbetha = 0
                dpsi = dx[2]
                dphi = 0
            return [dxpos, dypos, dbetha, dpsi, dphi]

    def carModelOneLaneSimple(self,x,t0):
        L = self.param.L
        Xpos = x[0]
        Ypos = x[1]
        psi = x[2]
        u = self.carInput(t0)
        v = u[0]
        delta = u[1]
        dx = np.array([np.cos(psi), np.sin(psi), np.tan(delta)/L])*v
        return dx



    def changeLaneTrajectory(self, Vsoll, t,right =True):
        if right:
            W=self.W
        else:
            W=-self.W

#        xsoll = 2*(Vsoll*self.T-D)*(t/self.T)**3-3*(Vsoll*self.T-D)*(t/self.T)**2+Vsoll*t
#        ysoll = -2*self.W*(t/self.T)**3+3*self.W*(t/self.T)**2
#        dxsoll = 6*(Vsoll*self.T-D)*(t**2/self.T**3)-6*(Vsoll*self.T-D)*(t/self.T**2)+Vsoll
#        dysoll = -6*self.W*(t**2/self.T**3)+6*self.W*(t/self.T**2)
#        ddxsoll = 12*(Vsoll*self.T-D)*(t/self.T**3)-6*(Vsoll*self.T-D)/self.T**2
#        ddysoll = -12*self.W*(t/self.T**3)+6*self.W/self.T**2

        a3 = -2*W/self.T**3
        a2 = 3*W/self.T**2
        if t<=self.T*0.8:
            xsoll = Vsoll*t
            ysoll = a3*t**3+a2*t**2
            dxsoll = Vsoll
            dysoll = 3*a3*t**2+2*a2*t
            ddxsoll = 0
            ddysoll = 6*a3*t+2*a2
        else:
            xsoll = Vsoll*t
            ysoll = W
            dxsoll = Vsoll
            dysoll = 0
            ddxsoll = 0
            ddysoll = 0
        p = np.array([xsoll, ysoll])
        dp = np.array([dxsoll, dysoll])
        ddp = np.array([ddxsoll, ddysoll])
        return p, dp, ddp


    def carInput(self, t, Vsoll=None,right =True):
        if not Vsoll:
            Vsoll = self.Vsoll

        p,dp,ddp = self.changeLaneTrajectory(Vsoll,t,right)
        v,delta = self.invModel(p,dp,ddp)
        if delta > self.degToRad(29):
            delta = self.degToRad(29)
        elif delta < self.degToRad(-29):
            delta = self.degToRad(-29)
        return np.array([v, delta]) # need maping to digitial values


    def invModel(self,p, dp, ddp):
        # dxsoll**2+dysoll**2 unequal 0
        # p is the desired trajectory dp ddp are derivative w.r.t. time
        L=self.param.L
        dxsoll = dp[0]
        dysoll = dp[1]
        ddxsoll = ddp[0]
        ddysoll = ddp[1]
        psi = np.arctan(dysoll/dxsoll)
        dpsi = (dxsoll*ddysoll-dysoll*ddxsoll)/(dxsoll**2+dysoll**2)
        v = dxsoll*np.cos(psi)+dysoll*np.sin(psi)
        delta = np.arctan(L*dpsi/v)
        return v, delta

    def radToDeg(self,x):
        return x/np.pi*180


    def degToRad(self,x):
        return x*np.pi/180
