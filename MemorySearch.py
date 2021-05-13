import numpy as np
import probCMR as CMR2_simple
class MemorySearch:
    def __init__(self, ll):
        num = 1 # examine only one list
        self.ll= ll # list length
        data = [[int(j+1)+k*ll for j in range(ll)] for k in range(num)]
        np.savetxt('datafile/K02_temp_data_structure.txt',data,delimiter=',')
        lists = [int(i+1) for i in range(num)]
        np.savetxt('datafile/K02_temp_list_ids.txt',lists)

        LSA_path = '/mnt/bucket/people/qiongz/optimalmemory/pyCMR2/K02_files/K02_LSA.txt'
        data_path = '/mnt/bucket/people/qiongz/CMRversions/datafile/K02_temp_data_structure.txt'    
        self.LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)        
        data_pres = np.loadtxt(data_path, delimiter=',')
        self.data_pres = np.reshape(data_pres, (1, ll))

        self.param_dict = {

            'beta_enc':  0.7887626184661226,           # rate of context drift during encoding
            'beta_rec':  0.49104864172027485,           # rate of context drift during recall
            'beta_rec_post': 1,      # rate of context drift between lists
                                            # (i.e., post-recall)

            'gamma_fc': 0.4024001271645564,  # learning rate, feature-to-context
            'gamma_cf': 1,  # learning rate, context-to-feature
            'scale_fc': 1 - 0.4024001271645564,
            'scale_cf': 0,


            's_cf': 0.0,       # scales influence of semantic similarity
                                    # on M_CF matrix

            's_fc': 0.0,            # scales influence of semantic similarity
                                    # on M_FC matrix.
                                    # s_fc is first implemented in
                                    # Healey et al. 2016;
                                    # set to 0.0 for prior papers.

            'phi_s': 4.661547054594787,      # primacy parameter
            'phi_d': 2.738934338758688,      # primacy parameter


            'epsilon_s': 0.0,      # baseline activiation for stopping probability 
            'epsilon_d': 2.723826426356652,        # scale parameter for stopping probability 

            'k':  5.380182482069175,        # scale parameter in luce choice rule during recall

            # parameters specific to optimal CMR:
            'primacy': 0.0,
            'enc_rate': 1.0,

        }
        
        


    def reset(self):
        # create CMR2 object
        self.CMR = CMR2_simple.CMR2(
        recall_mode=0, params=self.param_dict,
        LSA_mat=self.LSA_mat, pres_sheet = self.data_pres, rec_sheet = self.data_pres)

        # layer LSA cos theta values onto the weight matrices
        self.CMR.create_semantic_structure()
        
        # nothing recalled yet
        self.rec_item = -1
        
        # complete encoding the list
        self.CMR.present_list()
    
        # recall session
        self.CMR.recall_start()
        
        observation = np.concatenate((self.CMR.M_FC_tem.flatten(), self.CMR.M_CF_tem.flatten(),self.CMR.M_CF_sem.flatten(),self.CMR.c_net.T[0],self.CMR.torecall[0],[self.rec_item]), axis=0)
                
        return observation
    
    
                
    def step(self, action):
        while self.rec_item is not None:
            # start recall
            self.rec_item = self.CMR.recall_step(action[0],action[1])
            observation = np.concatenate((self.CMR.M_FC_tem.flatten(), self.CMR.M_CF_tem.flatten(),self.CMR.M_CF_sem.flatten(),self.CMR.c_net.T[0],self.CMR.torecall[0],[self.rec_item]), axis=0)
 
            print("The just-recalled item is {}".format(self.rec_item))
            remain = [i for i in range(self.ll) if self.CMR.torecall[0][i]>0]
            print("Remaining items are {}".format(remain))  

            if self.rec_item is not None:
                return observation, 1, 0
            else:
                return observation, 0, 1
