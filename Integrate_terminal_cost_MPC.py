import numpy as np
import casadi as ca

from tensorflow_template import get_critic, tf2np, np2tf


from casadi.casadi import SX, MX, DM
from casadi import fmax, Function, transpose

class OcpSinglePendulum:

    def __init__(self, dt, w_u, w_v, u_min=None, u_max=None, x_min = None, x_max = None, v_min = None, v_max = None):
        # Load weight for model
        self.V = get_critic(nx=1)
        self.V.load_weights("thuongdc.h5")

        self.params = []
        for idx, param in enumerate(self.V.get_weights()):
            if len(param.shape) == 1:
                param = param.reshape(1,-1)
            else:
                param = param.T
            self.params.append(MX(DM(param.tolist())))

        self.dt = dt
        self.w_u = w_u
        self.w_v = w_v
        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max
        self.v_min = v_min
        self.v_max = v_max

        self.m = 1
        self.l = 1
        self.g = 9.8

    def get_value(self, x):

        for idx, param in enumerate(self.params):

            if idx %2 == 0:
                x = param @ x
            else:
                x = x + transpose(param)

                if idx != len(self.params): x = fmax(0, x)
        # x = x * 100
        return x

    def solve(self, x_init,u_init, N, x_des, X_guess=None, U_guess=None):
        self.opti = ca.Opti()
        self.x = self.opti.variable(N+1)
        self.v = self.opti.variable(N+1)
        self.u = self.opti.variable(N)
        self.X = ca.horzcat(self.x,self.v)
        
        x = self.X
        u = self.u

        if(X_guess is not None):
            for i in range(N+1):
                self.opti.set_initial(x[i,0], X_guess[i,:])
        else:
            for i in range(N+1):
                self.opti.set_initial(x[i,0], x_init)
        if(U_guess is not None):
            for i in range(N):
                self.opti.set_initial(u[i], U_guess[i,:])
        # else:
        #     for i in range(N):
        #         self.opti.set_initial(u[i],u_init)

        self.cost = 0
        self.running_costs = [None,]*(N+1)

        # Running cost
        for i in range(N):
            self.running_costs[i] = self.w_v* (x[i,1] - x_des[1]**2)
            if(i<N-1):
                self.running_costs[i] += self.w_u * u[i]*u[i]
            self.cost += self.running_costs[i] 
        
        #Terminal cost
        terminal_cost = self.get_value(x[-1])
        self.running_costs[-1] = terminal_cost
        self.cost += terminal_cost
        
        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to(x[i+1,0] ==  self.dt * x[i,1])
            self.opti.subject_to(x[i+1,1] ==  self.dt * (u[i] + self.m * self.g * self.l * ca.sin(x[i,0])) / (self.m * self.l*self.l))
            #self.opti.subject_to( x[i]==x[i-1] + self.dt*u[i-1] )
        if(self.u_min is not None and self.u_max is not None):
            for i in range(N):
                self.opti.subject_to( self.opti.bounded(self.u_min, u[i], self.u_max) )

        if(self.v_min is not None and self.v_max is not None):
            for i in range(N+1):
                self.opti.subject_to( self.opti.bounded(self.v_min, x[i,1], self.v_max))

        self.opti.subject_to(x[0,0]==x_init)
        self.opti.subject_to(u[0]==u_init)

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    N = 100       # horizon size
    dt = 1e-2        # time step
    #x_init = 1.5 # initial state
    w_u = 1e-1    # weight for control input
    w_v = 1    #weight for velocity cost

    # u_min = -1      # min control input
    # u_max = 1       # max control input

    nq = 1  #number of joint position
    nv = 1  #number of joint velocity
    nu = 1  #number of control

    lowerPositionLimit = 3/4*np.pi      # min joint position
    upperPositionLimit = 5/4*np.pi      # max joint position
    upperVelocityLimit = 10             # min joint velocity
    lowerVelocityLimit = -10            # min joint velocity
    lowerControlBound    = -9.81    # lower bound joint torque
    upperControlBound    = 9.81       # upper bound joint torque

    x_des_final = np.array([0,0])  #final desired joint velocity

    DATA_FOLDER = 'data/'     # your data folder name
    DATA_FILE_NAME = 'warm_start' # your data file name
    save_warm_start = 0
    use_warm_start = 0
    if use_warm_start:
        INITIAL_GUESS_FILE = DATA_FILE_NAME
    else:
        INITIAL_GUESS_FILE = None
    
    n = nq + nv #state size
    m = nu  #control size



    is_plot = False

    data = []
    # X = np.linspace(-2.2, 2.0, 100)

    #Start MPC
    q0 = np.pi + np.pi/8 # Start at fixed state
    #X = np.random.uniform(lowerPositionLimit, upperPositionLimit, size =(n_ics, n))
    ocp = OcpSinglePendulum(dt, w_u, w_v, lowerControlBound, upperControlBound, lowerPositionLimit,upperPositionLimit,lowerVelocityLimit,upperVelocityLimit)


    U = np.zeros((N,m))
    if(INITIAL_GUESS_FILE is None):
            #use u that compensate gravity
        u0 = 9.81*np.sin(q0)
        for j in range(N):
            U[j,:] = u0
    else:
        print("Load initial guess from", INITIAL_GUESS_FILE)
        data = np.load(DATA_FOLDER + INITIAL_GUESS_FILE+'.npz')
        U = data['u']
    print('u',U)
    x_opt = []
    for i in range(N+1):
        sol = ocp.solve(q0,u0, N, x_des_final)
        x_opt.append(sol.value(ocp.x[:,0]))
        u_opt = sol.value(ocp.u[0])
        u_res = u_opt
        next_states_pred = sol.value(ocp.x[:,0])
        print(sol.value(ocp.u))
        #t = [i*dt for i in range(N+1)]
        print('ok', u0)
        i = i+1
        print('count',i)
    # print('x',x_opt)
    #print('u',u)
    #print('t',t)
        
    # plt.plot(t, x_opt, 'xr',label='state x')
    # #plt.stairs(u,t)
    # plt.xlabel("Time")
    # plt.legend()
    # plt.show()
        #costs = [sol.value(ocp.cost[0], [ocp.x==x_val]) for x_val in X[:,0]]
        #opt_cost_J = float(np.min(costs))
        #data.append({"x0":q0, "j_opt":opt_cost_J})
    if is_plot:
        plt.plot(X, costs)
        for i in range(N+1):
            plt.plot(sol.value(ocp.x[i]), sol.value(ocp.running_costs[i]), 
                            'xr', label='x_'+str(i))
        plt.legend()
        plt.show()
    
    # json.dump(data, open("valid.json", "w"))
    # df = pd.DataFrame(data)

    # data_test = json.load(open("test.json"))
    # df_test = pd.DataFrame(data_test)
    # print(df_test)
    # df.index = df["x0"]
    # df_test.index = df_test["x0"]

    # df_test.columns = ["x0", "y_val"]

    # df=df.join(df_test, lsuffix="_l", rsuffix="_r")

    # df.loc[:, "y_dif"] = df.loc[:, "y_val"] - df.loc[:, "j_opt"]
    # print(df)
    # print("error:      ", df["y_dif"].unique())


    # # load json file:
    
    # print(data)