from remesher_tong import RemesherTong
import msu

def grid_search(remesh_function, params_dictinary, savefile='./gridsearch.txt'):
    """
    :param remesher:
    :param params:
    :return:
    """
    param_enumeration = []
    initial_key = list(params_dictinary.keys())[0]

    for v in params_dictinary.pop(initial_key):
        param_enumeration.append({initial_key:v})

    for param in params_dictinary:
        if len(params_dictinary[param]) > 0:
            new_param_enumeration = []
            for x in param_enumeration:
                for v in params_dictinary[param]:
                    y = x.copy()
                    y[param] = v
                    new_param_enumeration.append(y)
            param_enumeration = new_param_enumeration
    results = []
    for p in param_enumeration:
        f = open(savefile, 'a')
        p_frozen = frozenset(p.items())
        res = None
        try:
            res = remesh_function(**p)
            results.append({p_frozen: res})
        except Exception as e:
            #print(f"warning: parameter set {p} causes error: "+str(e))
            res = str(e)
        print(f"{res} | {p}")
        f.write(f"{res} | {p}\n")
        f.close()
    return results


if __name__ == '__main__':

    case = '../cases/bunny_fixed.dat'

    params = {
        'name_in': [case],
        'name_out':['../res_tong.dat'],
        'steps': [10],
        'normal_smoothing_steps': [],#[20,30],
        'normal_smoothing_s': [5,10,15],
        'normal_smoothing_k': [0.05, 0.15, 0.3],
        'height_smoothing_steps':[],# [20, 30, 40],
        'time_step_fraction_k': [],
        'null_space_smoothing_steps': [0, 5, 10],
        'height_smoothing_alpha': [0.3, 0.5, 0.8],
        'height_smoothing_b': [0.6, 0.4, 0.2],
        'threshold_for_null_space': [0.03, 0.1, 0.3]
    }

    results = grid_search(RemesherTong(tracking_evolution=False).remesh, params, savefile="../gridsearch.txt")
    for r in results:
        print(r)