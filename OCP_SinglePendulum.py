import numpy as np
import casadi as ca
from train import get_critic, tf2np, np2tf
from casadi.casadi import SX, MX, DM
from casadi import fmax, Function, transpose


class OcpSinglePendulum:

    def __init__(self, dt, w_u,w_x, w_v, u_min=None, u_max=None, x_min = None, x_max = None, v_min = None, v_max = None):
        self.dt = dt
        self.w_u = w_u
        self.w_x = w_x
        self.w_v = w_v
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max

        self.g = 9.8
    
    def objective_cost(self,N,x,x_des,u):
        cost = 0
        running_costs = [None,]*(N+1)
        for i in range(N+1):
            """
            running cost function = 
            """
            running_costs[i] = self.w_x * (x[i,0] - x_des[0])**2 + self.w_v * (x[i,1] - x_des[1])**2 
            if(i<N):
                running_costs[i] += self.w_u* u[i]*u[i]
            cost += running_costs[i] 
        cost += self.w_v*(x[N,1])**2
        return cost
    
    def OCP_setup(self, x_init, N, x_des, X_guess=None, U_guess=None):
        self.opti = ca.Opti()
        self.x = self.opti.variable(N+1)
        self.v = self.opti.variable(N+1)
        self.X = ca.horzcat(self.x,self.v)
        self.u = self.opti.variable(N)
        x = self.X
        u = self.u
        if(X_guess is not None):
            for i in range(N+1):
                self.opti.set_initial(x[i,:], X_guess[i,:])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i,:], x_init)
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i,:][0])

        self.cost = self.objective_cost(N,x,x_des,u)

        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to(x[i+1,0] == x[i,0] + self.dt * x[i,1])
            self.opti.subject_to(x[i+1,1] == x[i,1] + self.dt * (u[i] + self.g * ca.sin(x[i,0])))
            #self.opti.subject_to( x[i+1]==x[i] + self.dt*u[i] )
        if(self.u_min is not None and self.u_max is not None):
            for i in range(N):
                self.opti.subject_to( self.opti.bounded(self.u_min, u[i], self.u_max))
        # if(self.x_min is not None and self.x_max is not None):
        #     for i in range(N+1):
        #         self.opti.subject_to( self.opti.bounded(self.x_min, x[i,0], self.x_max))
        if(self.v_min is not None and self.v_max is not None):
            for i in range(N+1):
                self.opti.subject_to( self.opti.bounded(self.v_min, x[i,1], self.v_max))
        self.opti.subject_to(x[0,:].T== x_init)
        self.opti.subject_to(x[-1,:] == x[N-1,:]) #add hard constraints on the final state
        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

    def solve(self):
        return self.opti.solve()

class OcpSinglePendulumWithNNCost(OcpSinglePendulum):
    def __init__(self, dt, w_u,w_x, w_v, u_min=None, u_max=None, x_min = None, x_max = None, v_min = None, v_max = None):
        super().__init__(dt, w_u, w_x, w_v, u_min, u_max, x_min, x_max, v_min, v_max)
        self.V = get_critic(nx=2)
        self.V.load_weights("thuongdc.h5")

        self.params = []
        for idx, param in enumerate(self.V.get_weights()):
            if len(param.shape) == 1:
                param = param.reshape(1,-1)
            else:
                param = param.T
            self.params.append(MX(DM(param.tolist())))

    def get_value(self, x):
        for idx, param in enumerate(self.params):
                # print('iteration',idx)
                # print('shape of params',param.shape)
                # print('shape of input', x.shape)
            if idx %2 == 0:
                x = param @ x
            else:
                x = x + transpose(param)

                if idx != len(self.params): x = fmax(0, x)
            # x = x * 100
        return x

    def objective_cost(self,N,x,x_des,u):
        cost = 0
        running_costs = [None,]*(N+1)
        for i in range(N+1):
            """
            running cost function = 
            """
            running_costs[i] = self.w_x * (x[i,0] - x_des[0])**2 + self.w_v * (x[i,1] - x_des[1])**2 
            if(i<N):
                running_costs[i] += self.w_u* u[i]*u[i]
            cost += running_costs[i] 
        cost += self.w_v*(x[N,1])**2
        #Terminal cost
        # terminal_cost = self.get_value(x[-1,:].T)
        # running_costs[-1] = terminal_cost
        # cost += terminal_cost
        return cost
    