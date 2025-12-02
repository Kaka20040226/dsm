import torch,json
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from PPO import network, point # ppo method
from xtmqn import network, point # dqn method
# from xtmqnorg import network, point # dqn method
# from xtmqn2 import network, point # dqn method
# from basic_dqn import network, point # basic dqn method
# from no_attention_dqn import network, point # no attention dqn method
# from xtmqn_no_expert import network, point # xtmqn no expert method
from copy import deepcopy

def plot_metrics(draw, objvs, losslist, outdir="plots", smooth_window=5, dpi=150):
    """Save nicer, smoothed loss and reward/objective plots to `outdir`.

    Improvements:
    - optional moving-average smoothing (`smooth_window`)
    - nicer default colors, grid and higher DPI
    - consistent axes formatting
    """
    os.makedirs(outdir, exist_ok=True)

    def moving_average(y, w):
        y = np.asarray(y, dtype=float)
        if w <= 1 or len(y) < w:
            return y, np.arange(len(y))
        ma = np.convolve(y, np.ones(w, dtype=float) / w, mode='valid')
        # align x to the center of the window
        offset = (w - 1) // 2
        xs = np.arange(offset, offset + len(ma))
        return ma, xs

    plt.style.use('seaborn-v0_8')
    colors = plt.cm.tab10

    # Per-network plots from draw
    try:
        for idx, (k, v) in enumerate(draw.items()):
            losses = v[0] if len(v) > 0 else []
            obj1 = v[1] if len(v) > 1 else []
            obj2 = v[2] if len(v) > 2 else []

            # Loss plot (raw + smoothed)
            if len(losses) > 0:
                plt.figure(figsize=(10, 4), dpi=dpi)
                x = np.arange(len(losses))
                plt.plot(x, losses, color=colors(0), alpha=0.25, linewidth=1, label='loss (raw)')
                ma, xs = moving_average(losses, smooth_window)
                if len(ma) > 0:
                    plt.plot(xs, ma, color=colors(1), linewidth=2, label=f'loss (ma w={smooth_window})')
                plt.xlabel('training steps')
                plt.ylabel('loss')
                plt.grid(alpha=0.4)
                plt.legend()
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f'loss.png'))
                plt.close()

            # Reward/objective plot (obj1, obj2 and sum)
            if len(obj1) > 0 or len(obj2) > 0:
                plt.figure(figsize=(10, 4), dpi=dpi)
                plotted = False
                if len(obj1) > 0:
                    x1 = np.arange(len(obj1))
                    plt.plot(x1, obj1, color=colors(2), alpha=0.25, linewidth=1, label='obj1 (raw)')
                    ma1, xs1 = moving_average(obj1, smooth_window)
                    if len(ma1) > 0:
                        plt.plot(xs1, ma1, color=colors(3), linewidth=2, label=f'obj1 (ma w={smooth_window})')
                    plotted = True
                if len(obj2) > 0:
                    x2 = np.arange(len(obj2))
                    plt.plot(x2, obj2, color=colors(4), alpha=0.25, linewidth=1, label='obj2 (raw)')
                    ma2, xs2 = moving_average(obj2, smooth_window)
                    if len(ma2) > 0:
                        plt.plot(xs2, ma2, color=colors(5), linewidth=2, label=f'obj2 (ma w={smooth_window})')
                    plotted = True
                # plot sum when lengths match
                if len(obj1) == len(obj2) and len(obj1) > 0:
                    summed = [a + b for a, b in zip(obj1, obj2)]
                    ma_sum, xs_sum = moving_average(summed, smooth_window)
                    if len(ma_sum) > 0:
                        plt.plot(xs_sum, ma_sum, color=colors(6), linewidth=2.5, label='obj_sum (ma)')
                        plotted = True
                if plotted:
                    plt.xlabel('Episode')
                    plt.ylabel('Total reward')
                    plt.grid(alpha=0.35)
                    plt.legend()
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, f'reward.png'))
                    plt.close()
    except Exception as e:
        print(f"plot_metrics: failed to plot per-network figures: {e}")

    # Aggregate plots
    try:
        if objvs:
            vals = [x[0] for x in objvs]
            plt.figure(figsize=(10, 4), dpi=dpi)
            x = np.arange(len(vals))
            plt.plot(x, vals, color=colors(7), alpha=0.2, linewidth=1, label='total_obj (raw)')
            ma, xs = moving_average(vals, smooth_window)
            if len(ma) > 0:
                plt.plot(xs, ma, color=colors(8 % 10), linewidth=2, label=f'total_obj (ma w={smooth_window})')
            plt.xlabel('episode')
            plt.ylabel('total reward')
            plt.grid(alpha=0.35)
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'total_obj.png'))
            plt.close()
    except Exception as e:
        print(f"plot_metrics: failed to plot aggregate objvs: {e}")

    try:
        # Flatten losslist if it's nested
        flat = []
        for item in (losslist or []):
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                for x in item:
                    try:
                        flat.append(float(x))
                    except:
                        pass
            else:
                try:
                    flat.append(float(item))
                except:
                    pass

        if flat:
            plt.figure(figsize=(10, 4), dpi=dpi)
            x = np.arange(len(flat))
            plt.plot(x, flat, color=colors(0), alpha=0.2, linewidth=1, label='train_loss_all (raw)')
            ma, xs = moving_average(flat, smooth_window)
            if len(ma) > 0:
                plt.plot(xs, ma, color=colors(1), linewidth=2, label=f'train_loss_all (ma w={smooth_window})')
            plt.xlabel('training step')
            plt.ylabel('loss')
            plt.title('Training Loss (flattened)')
            plt.grid(alpha=0.35)
            plt.legend()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'train_loss_all.png'))
            plt.close()
    except Exception as e:
        print(f"plot_metrics: failed to plot losslist: {e}")
    print(f"Saved plots to {outdir}")
import numpy as np
import json

device = "cuda"
datafile = "dsm-main\data_60_30.json"

def process_bar(i, epoch, loss, taskN, objv, t, objs):
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("T"+"|"*int(i/epoch*100+0.5)+" "*int((epoch-i)/epoch*100+0.5)+f"||{i}/{epoch} loss:{loss:7.5} Time:{t:.5f} objvalue : {float(objs[0]+objs[1]):7.5} ")

def main(epoch, det):
    
    
    taskN = det["taskNum"]
    providerN = det["providerNum"]
    edges = det["edges"]
    
    deadlines = np.array(det["taskdeadlines"])
    budgets = np.array(det["taskbudgets"])
    Rs = np.array(det["taskResources"])
    abilities = np.array(det["taskabilities"])
    providerRep = np.array(det["providerRep"])
    cost = np.array(det["taskCost"])
    # epoch = 400
    netsNum = 1
    # netsNum = 0
    cnt = 10
    
    nets = []
    draw = {
    }
    bestSolution = {
    }
    for i in range(netsNum):
        net_new = network(deadlines, budgets, Rs, abilities, cost,\
                          providerN,(providerN+1)*taskN,providerN,taskN, edges,device=device)
        # if os.path.exists(f"model_{net_new.task_num}_{net_new.provider_num}_ppo.pth"):
        #     net_new.model.load_state_dict(torch.load(f"model_{net_new.task_num}_{net_new.provider_num}_ppo.pth"))
        net_new.taskTime = det["taskTime"]
        net_new.rep = det["providerRep"]
        net_new.deadline = det["deadline"]
        net_new.deadlines = det["taskdeadlines"]
        net_new.providerAbility = det["providerAbility"]
        net_new.providerL = torch.tensor(det["providerL"]).to(device)
        net_new.providerPrice = det["providerPrice"]
        net_new.providerReliability = det["providerReliability"]
        net_new.providerEnergyCost = det["providerEnergyCost"]
        net_new.budget = det["budget"]
        net_new.param = det["providerParam"]
        net_new.andor = det["andor"]
        net_new.budgets = det["taskbudgets"]
        net_new.critical_path = -1
        net_new.paths = [0 for i in range(taskN)]
        # net_new.model.load_state_dict(torch.load(f"model_{net_new.task_num}_{net_new.provider_num}_ppo.pth"))
        net_new.charData = torch.cat([
            torch.tensor(net_new.deadlines, dtype=torch.float32).flatten().to(device),
            torch.tensor([net_new.budget], dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.rep, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerL, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerPrice, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerReliability, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerEnergyCost, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.param, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.taskTime, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerAbility, dtype=torch.float32).flatten().to(device)
        ], dim=0).to(device).unsqueeze(0)
        startp = point()
        startp.loc = -1
        startp.children = [net_new.points[0]]
        net_new.init_points()
        startp.finished = True
        startp.L = 0
        startp.hc = 0
        net_new.set_beginning(startp)
        nets.append((net_new, startp))
        draw[i] = [[],[],[],[]]
        bestSolution[i] = [(-1*net_new.M*100*taskN,0,0),-1*net_new.M*100*taskN,None]                  
    import time
    pretime = time.time()
    preTime = time.time()
    objs = 0
    objvs = []
    losslist = []
    
    print(f"Start training")
    for j in range(epoch):
        if j % (100//(taskN**2)+1) == 0:
            pretime = time.time()
        for i, dt in enumerate(nets):
            net = dt[0]
            startp = dt[1]
            # print('-'*10+f"net {i}"+'-'*10)
            param = [x.clone for x in net.model.parameters()]
            loss, losses = net.training_step(startp,i)
            losslist.append(losses)
            if not net.critical_path == -1:
                draw[i][0].append((loss).item())
            else:
                draw[i][0].append(0)

            print(loss)
            if (time.time()-pretime > 20 or True) and (loss != 0):
                startp.finished = True
                net.init_net()
                pretime = time.time()
                calculated, x = net.search(startp)
                objvt, objs = net.objv(x[-1])
                draw[i][1].append(objs[0].item())
                draw[i][2].append(objs[1].item())
                draw[i][3].append(net.critical_path)
                # torch.save((net.model.state_dict(),loss), "test.pth")
                print(f"epoch:{j}/{epoch} loss:{loss} objv1:{objs[0]} objv2:{objs[1]}")
                if bestSolution[i][0][0] < objs[0].item() + objs[1].item():
                    # net.critical_path = -1
                    # net.traceCriticalPath(startp, 0, [])
                    bestSolution[i] = [(objs[0].item()+objs[1].item(),objs[0].item(), objs[1].item()), objs, deepcopy(net.points), j, None , net.critical_path, net.cost]
                    import pickle
                    pickle.dump(net.points, open(f"points_{net.task_num}_{net.provider_num}_ppo.pkl", "wb"))
                    torch.save(net.model.state_dict(), f"model_{net.task_num}_{net.provider_num}_ppo.pth")
                objvs.append((objs[0].item()+objs[1].item(), objs[0].item(), objs[1].item()))
                losses.append(loss.item())
            # process_bar(j, epoch, loss, taskN, objv)
            startp.finished = True
            net.init_net()
        
        if j % (100//(taskN**2)+1) == 0:
            # torch.save((net.model.state_dict(),loss), "test.pth") 
            json.dump(draw, open("draw.json","w"), indent=4)
            process_bar(j, epoch, loss, taskN, objs, time.time()-pretime, objs)
            print(f"Best Solution: {bestSolution[i][0]}")    
    print(time.time()-preTime)
    json.dump(draw, open("draw_attnet.json","w"), indent=4)
    torch.save(losslist, "losslist.pth")
    pickle.dump(losses, open("losses.pkl", "wb"))
    pickle.dump(objvs, open("objvs.pkl", "wb"))
    print("loss and rewards are saved in losses.pkl and objvs.pkl")
    try:
        plot_metrics(draw, objvs, losslist, outdir="plots")
    except Exception as e:
        print(f"Failed to generate plots: {e}")
    # for det in nets:
    #     net = det[0]
    #     startp = det[1]
    #     net.init_net()
    #     startp.finished = False
    #     calculated, x = net.search(startp)
    #     obj, objs = net.objv(x[-1])
    #     print(objs[0]+objs[1], objs)
    #     print(net.critical_path)
    #     for p in net.points:
    #         print(p.loc, p.provider, p.finished)
    #     print("----"*3)
    
    print("-"*50)
    print(" "*18+"Best Solution")
    print("-"*50)
    
    # bestSolutionData = {}
    # for b in bestSolution.keys():
    #     bestSolutionData[b] = {
    #         "objv": bestSolution[b][0],
    #         "objs": [x.item() for x in bestSolution[b][1]],
    #         "points": [[p.loc, p.provider, p.finished] for p in bestSolution[b][2]],
    #         "epoch": bestSolution[b][3],
    #         "path": bestSolution[b][4],
    #         "critical_path": bestSolution[b][5],
    #         "cost": bestSolution[b][6]
    #     }
    #     print(bestSolution[b][0], bestSolution[b][1])
    #     for p in bestSolution[b][2]:
    #         print(p.loc, p.provider, p.finished)
    #     print(f"epoch:{bestSolution[b][3]}")
    #     print(f"critical path:{bestSolution[b][5]}")
    #     print(f"cost:{bestSolution[b][6]}")
    #     print(f"path:{bestSolution[b][4]}")
    #     print("----"*3)

    # json.dump(bestSolutionData, open(f"bestSolution{datafile.split('.')[0]}_{network.name}.json","w"), indent=4)

def run(epoch, det):
    
    
    taskN = det["taskNum"]
    providerN = det["providerNum"]
    edges = det["edges"]
    
    deadlines = np.array(det["taskdeadlines"])
    budgets = np.array(det["taskbudgets"])
    Rs = np.array(det["taskResources"])
    abilities = np.array(det["taskabilities"])
    providerRep = np.array(det["providerRep"])
    cost = np.array(det["taskCost"])

    # epoch = 400
    netsNum = 1
    # netsNum = 0
    cnt = 10
    # net.model.load_state_dict(torch.load("test.pth")[0])
    nets = []
    draw = {
    }
    bestSolution = {
    }
    for i in range(netsNum):
        net_new = network(deadlines, budgets, Rs, abilities, cost,\
                          providerN,(providerN+1)*taskN,providerN,taskN, edges,device=device)
        if os.path.exists(f"model_{net_new.task_num}_{net_new.provider_num}_ppo.pth"):
            net_new.model.load_state_dict(torch.load(f"model_{net_new.task_num}_{net_new.provider_num}_ppo.pth"))
        net_new.taskTime = det["taskTime"]
        net_new.rep = det["providerRep"]
        net_new.deadline = det["deadline"]
        net_new.deadlines = det["taskdeadlines"]
        net_new.providerAbility = det["providerAbility"]
        net_new.providerL = torch.tensor(det["providerL"]).to(device)
        net_new.providerPrice = det["providerPrice"]
        net_new.providerReliability = det["providerReliability"]
        net_new.providerEnergyCost = det["providerEnergyCost"]
        net_new.budget = det["budget"]
        net_new.param = det["providerParam"]
        net_new.andor = det["andor"]
        net_new.critical_path = -1
        net_new.paths = [0 for i in range(taskN)]
        # net_new.model.load_state_dict(torch.load("test.pth")[0])
        net_new.charData = torch.cat([
            torch.tensor(net_new.deadlines, dtype=torch.float32).flatten().to(device),
            torch.tensor([net_new.budget], dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.rep, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerL, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerPrice, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerReliability, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerEnergyCost, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.param, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.taskTime, dtype=torch.float32).flatten().to(device),
            torch.tensor(net_new.providerAbility, dtype=torch.float32).flatten().to(device)
        ], dim=0).to(device).unsqueeze(0)
        startp = point()
        startp.loc = -1
        startp.children = [net_new.points[0]]
        net_new.init_points()
        startp.finished = True
        startp.L = 0
        startp.hc = 0
        net_new.set_beginning(startp)
        nets.append((net_new, startp))
        draw[i] = [[],[],[],[]]
        bestSolution[i] = [(-1*net_new.M*100*taskN,0,0),-1*net_new.M*100*taskN,None]
    import time
    pretime = time.time()
    preTime = time.time()
    for j in range(epoch):
        if j % (100//(taskN**2)+1) == 0:
            pretime = time.time()
        for i, dt in enumerate(nets):
            net = dt[0]
            startp = dt[1]
            # print('-'*10+f"net {i}"+'-'*10)
            param = [x.clone for x in net.model.parameters()]
            _,_, objv, objs = net.run_step(startp,i)
            # print(loss/taskN, objv)
            print(f"objv1:{objs[0]} objv2:{objs[1]}")
            print(f"objv:{objv}")
            
    print(time.time()-preTime)
    json.dump(draw, open("draw_attnet.json","w"), indent=4)
    # for det in nets:
if __name__ == "__main__":
    torch.manual_seed(244)
    np.random.seed(4244)
    import os
    # for i in range(100):
    #     for _, __, ___ in os.walk("60_30"):
    #         for fil in ___:
    #             if fil.endswith(".json"):
    #                 datafile = os.path.join(_, fil)
    #                 print(f"Processing {datafile}")
    #                 det = json.load(open(datafile, "r"))
    #                 main(10, det)
    
    # datafile = "data_60_30.json"
    # datafile = "60_30/data_60_30_13096.json"
    # datafile = "5_3/data_5_3_0.json"
    
    print(f"Processing {datafile}")

    det = json.load(open(datafile, "r"))
    main(800, det)
    # datafiles = [
    #     '60_30/data_60_30_15752.json',
    #     '60_30/data_60_30_25677.json',
    #     '60_30/data_60_30_26970.json',
    #     '60_30/data_60_30_27564.json',
    #     '60_30/data_60_30_44257.json',
    #     '60_30/data_60_30_51255.json',
    #     '60_30/data_60_30_52756.json',
    #     '60_30/data_60_30_60276.json',
    #     '60_30/data_60_30_70766.json',
    #     '60_30/data_60_30_83451.json',
    #     '60_30/data_60_30_94725.json',
    #     '60_30/data_60_30_96075.json',
    #     '60_30/data_60_30_97770.json',
    #     '60_30/data_60_30_98043.json',
    #     '60_30/data_60_30_122587.json',
    #     '60_30/data_60_30_124246.json',
    #     '60_30/data_60_30_142111.json',
    #     '60_30/data_60_30_144200.json',
    #     '60_30/data_60_30_144771.json',
    #     '60_30/data_60_30_152620.json',
    #     '60_30/data_60_30_159971.json',
    #     '60_30/data_60_30_160066.json',
    #     '60_30/data_60_30_161316.json',
    # ]
    # epoches = 1000
    # for epoch in range(epoches):
    #     for datafile in datafiles:
    #         print(f"Processing {datafile}")
    #         det = json.load(open(datafile, "r"))
    #         main(50, det)
    #     test_datafile = "60_30/data_60_30_13096.json"
    #     det = json.load(open(test_datafile, "r"))
    #     print(f"Testing {test_datafile}")
    #     main(1, det)
    # run(1, det)