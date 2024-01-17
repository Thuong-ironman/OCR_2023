import numpy as np
import casadi as ca
import A3_conf as conf
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
    def solve(self, x_init, N, x_des, X_guess=None, U_guess=None):
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
                self.opti.set_initial(u[i], U_guess[i,:])

        self.cost = 0
        self.running_costs = [None,]*(N+1)
        for i in range(N+1):
            self.running_costs[i] = self.w_x * (x[i,0] - x_des[0])**2 + self.w_v * (x[i,1] - x_des[1])**2
            if(i<N):
                self.running_costs[i] += self.w_u* u[i]*u[i]
            self.cost += self.running_costs[i] 
        
        self.opti.minimize(self.cost)

        for i in range(N):
            self.opti.subject_to(x[i+1,0] == self.dt * x[i,1])
            self.opti.subject_to(x[i+1,1] == self.dt * (u[i] + self.g * ca.sin(x[i,0])))
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

        # s_opts = {"max_iter": 100}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver("ipopt", opts) #, s_opts)

        return self.opti.solve()


if __name__=="__main__":
    import json
    import matplotlib.pyplot as plt
    N = conf.N        # horizon size
    dt = conf.dt        # time step

    w_u = conf.w_u    # weight for control input
    w_x = conf.w_x
    w_v = conf.w_v    # weight for velocity cost


    nq = conf.nq                              # number of joint position
    nv = conf.nv                              # number of joint velocity
    nu = conf.nu                              # number of control

    lowerPositionLimit = conf.lowerPositionLimit      # min joint position
    upperPositionLimit = conf.upperPositionLimit      # max joint position
    
    lowerVelocityLimit = conf.lowerVelocityLimit            # min joint velocity
    upperVelocityLimit = conf.upperVelocityLimit             # max joint velocity

    lowerControlBound    = conf.lowerControlBound    # lower bound joint torque
    upperControlBound    = conf.upperControlBound       # upper bound joint torque

    x_des_final = conf.x_des_final        # final desired joint position and velocity

    n = nq + nv #state size
    m = nu      #control size
    is_plot = False

    data = []
   #Start MPC
    n_ics =  200          # TODO number of initial states to be checked
    x_min = [lowerPositionLimit, lowerVelocityLimit]
    x_max = [upperPositionLimit, upperVelocityLimit]
    X = np.random.uniform(x_min,x_max, size=(n_ics,n)) 
    ocp = OcpSinglePendulum(dt, w_u,w_x, w_v, lowerControlBound, upperControlBound, lowerPositionLimit,upperPositionLimit,lowerVelocityLimit,upperVelocityLimit)
    for i in range(int(n_ics)):
        x0 = X[i,:] # initial state in each iteration X[0] is position, X[1] is velocity
        sol = ocp.solve(x0, N,x_des_final)
        #print("Optimal value of x:\n", sol.value(ocp.x))
        costs = [sol.value(ocp.cost[0], [ocp.x==x_val]) for x_val in X[:,0]]
        opt_cost_J = float(np.sum(costs)) # t
        #print("Optimal cost J:\n", opt_cost_J)
        data.append({"x0":x0.tolist(), "j_opt":opt_cost_J}) 
        if is_plot:
            plt.plot(X, costs)
            for i in range(N+1):
                print('position',ocp.x.shape)
                plt.plot(sol.value(ocp.x[i]), sol.value(ocp.cost[i]), 
                        'xr', label='x_'+str(i))
            plt.legend()
            plt.show()
    print(data)
    with open('test.json', 'w') as json_file:
        json.dump(data, json_file) 
    # print(data)

    # load json file:
    data = json.load(open("test.json"))
    #print(data)