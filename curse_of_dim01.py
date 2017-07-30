import math
# demonstration of the curse of dimensionality
# 4 disks inside a box
# Where are the disks after 100 time steps?
from decimal import *
getcontext().prec = 18

# calculate time to collision with wall --------------------------------------
def wall_time(pos_a, vel_a, sigma):
    if vel_a > 0.0:
        del_t = (1.0 - sigma - pos_a) / vel_a
    elif vel_a < 0.0:
        del_t = (pos_a - sigma) / abs(vel_a)
    else:
        del_t = float('inf')
    return del_t
# calculate time to collsion with particle -----------------------------------
def pair_time(pos_a, vel_a, pos_b, vel_b, sigma):
    del_x = [pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]]
    del_x_sq = del_x[0] ** 2 + del_x[1] ** 2
    del_v = [vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]]
    del_v_sq = del_v[0] ** 2 + del_v[1] ** 2
    scal = del_v[0] * del_x[0] + del_v[1] * del_x[1]
    Upsilon = scal ** 2 - del_v_sq * ( del_x_sq - 4.0 * sigma **2)
    if Upsilon > 0.0 and scal < 0.0:
        del_t = - (scal + math.sqrt(Upsilon)) / del_v_sq
    else:
        del_t = float('inf')
    return del_t

# init positions and velocities for 4 particles ------------------------------
pos = [[0.250000, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
vel = [[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.1177]]
singles = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)] #(disk,direction) pairs
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] #how many distinct pairs?
# disk radius, start time, number of events ----------------------------------
sigma = 0.15
t = 0.0
n_events = 100

# Big event loop -------------------------------------------------------------
for event in range(n_events):
    # calculate time to next wall collision ----------------------------------
    wall_times = [wall_time(pos[k][l], vel[k][l], sigma) for k, l  in singles]
    # calculate time to next particle - to - particle collision --------------
    pair_times = [pair_time(pos[k], vel[k], pos[l], vel[l], sigma) for k, l in pairs]
    next_event = min(wall_times + pair_times)
    t += next_event
    # position of next event
    for k, l in singles: pos[k][l] += vel[k][l] * next_event 
    # check if wall collition
    if min(wall_times) < min(pair_times):
        collision_disk, direction = singles[wall_times.index(next_event)]
        vel[collision_disk][direction] *= -1.0 # do a reflection on wall collision
    else: 
        # do a pair collision
        a, b = pairs[pair_times.index(next_event)]
        del_x = [pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]]
        abs_x = math.sqrt(del_x[0] ** 2 + del_x[1] ** 2)
        e_perp = [c / abs_x for c in del_x] #unit direction vector
        del_v = [vel[b][0] - vel[a][0], vel[b][1] - vel[a][1]] #delta velocity vector
        scal = del_v[0] * e_perp[0] + del_v[1] * e_perp[1] # projection of velocity on unit collision direction
        for k in range(2): 
            vel[a][k] += e_perp[k] * scal #reflection of perpendicular velocity direction
            vel[b][k] -= e_perp[k] * scal 
    #print 'event', event
    #print 'time', t 
    #print 'wall', wall_times
    #print 'pair', pair_times
    #print 'pos', pos
    #print 'vel', vel

print('event', event)
print('time', t)
print('pos', pos)

#event 100
#time 6.36010393101
#pos [[0.7217884445958171, 0.85], [0.8375690373531577, 0.546793225810059], [0.316978840352578, 0.42753404775012427], [0.16624374157804678, 0.8010609870874347]]



#event 99
#time 5.24927442637
#pos [[0.4705864318188132, 0.402615284052826], [0.821924632254859, 0.20872855244333718], [0.15, 0.5623774002752693], [0.8157385959995493, 0.8466113698125248]]
#vel [[0.11057788296618187, -0.08725963925497054], [-0.3558922495473892, 0.5968325889562937], [0.49600019321793853, 0.5547307524894192], [-0.7791504032172888, 0.480338832871403]]


#event 1
#time 0.128205128205
#wall [2.857138095238095, 5.0, 0.1408450704225352, 3.3333333333333335, 0.43478260869565216, 0.7594936708860759, 0.12820512820512817, 0.8496176720475784]
#pair [inf, 0.2410756230423861, inf, inf, inf, inf]
#pos [[0.2769240769230769, 0.2653846153846154], [0.841025641025641, 0.27307692307692305], [0.2205128205128205, 0.6487179487179487], [0.85, 0.7650897435897436]]
#vel [[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [-0.78, 0.1177]]

#event 2
#time 0.140845070423
#wall [2.7289329670329674, 4.871794871794871, 0.012639942217406962, 3.205128205128205, 0.30657748049052397, 0.6312885426809477, 0.8974358974358974, 0.7214125438424503]
#pair [inf, 0.11287049483725801, inf, inf, inf, inf]
#pos [[0.2795784647887324, 0.2669014084507042], [0.85, 0.2753521126760563], [0.21760563380281692, 0.6387323943661972], [0.8401408450704225, 0.7665774647887323]]
#vel [[0.21, 0.12], [-0.71, 0.18], [-0.23, -0.79], [-0.78, 0.1177]]

#event 3
#time 0.241075623042
#wall [2.71629302481556, 4.859154929577465, 0.9859154929577465, 3.1924882629107985, 0.293937538273117, 0.6186486004635408, 0.8847959552184903, 0.7087726016250436]
#pair [0.2951787093240077, 0.10023055261985098, inf, inf, inf, inf]
#pos [[0.3006268808389011, 0.2789290747650863], [0.7788363076399057, 0.2933936121476295], [0.19455260670025118, 0.5595502577965149], [0.7619610140269387, 0.7783746008320888]]
#vel [[0.4559657292465651, -0.5307062573545285], [-0.71, 0.18], [-0.4759657292465651, -0.13929374264547156], [-0.78, 0.1177]]

#event 4
#time 0.334680275568
#wall [1.204856163354381, 0.2429386746027335, 0.8856849403378955, 3.0922577102909474, 0.09360465252566859, 2.940191354028704, 0.7845654025986394, 0.608542049005193]
#pair [0.18551835949217615, inf, inf, inf, inf, inf]
#pos [[0.3433073944886389, 0.22925249995221764], [0.712377004346681, 0.31024244960224984], [0.15, 0.5465117154171857], [0.6889493850569173, 0.7893918684343599]]
#vel [[0.4559657292465651, -0.5307062573545285], [-0.71, 0.18], [0.4759657292465651, -0.13929374264547156], [-0.78, 0.1177]]

#event5
#time 0.426593982535
#wall [1.1112515108287124, 0.1493340220770649, 0.7920802878122268, 2.9986530577652784, 1.4706941214193556, 2.846586701503036, 0.6909607500729709, 0.5149373964795246]
#pair [0.09191370696650769, inf, inf, 0.2571838526872258, inf, inf]
#pos [[0.3852168949133777, 0.1804733205284415], [0.6471182724004606, 0.32678691685622124], [0.19374777456406891, 0.5337087111734017], [0.6172566936230413, 0.8002101117443179]]
#vel [[-0.13005909563403395, -0.8580943776016878], [-0.12397517511940093, 0.5073881202471593], [0.4759657292465651, -0.13929374264547156], [-0.78, 0.1177]]
