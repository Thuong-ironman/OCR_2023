import numpy as np
import casadi as ca
import A3_conf as conf
from OCP_SinglePendulum import OcpSinglePendulum

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
        sol = ocp.OCP_setup(x0, N,x_des_final)
        sol = ocp.solve()
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