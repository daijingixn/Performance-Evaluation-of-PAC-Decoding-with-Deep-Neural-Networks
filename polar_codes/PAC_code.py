import polar_codes.polar_coding_exceptions as pcexc
import polar_codes.polar_coding_functions as pcfun
from polar_codes.rate_profile import rateprofile
import copy
import numpy as np
import csv
import math



class PACCode:
    """Represent constructing polar codes,
    encoding and decoding messages with polar codes"""

    def __init__(self, N, K, construct, dSNR, rprofile):
        if K >= N:
            raise pcexc.PCLengthError
        elif pcfun.np.log2(N) != int(pcfun.np.log2(N)):
            raise pcexc.PCLengthDivTwoError
        else:
            self.codeword_length = N
            self.log2_N = int(math.log2(N))
            self.nonfrozen_bits = K
            self.information_size = K
            self.designSNR = dSNR
            self.n = int(np.log2(self.codeword_length))
            #self.bitrev_indices = np.array([pcfun.bitreversed(j, self.n) for j in range(N)])
            self.bitrev_indices = [pcfun.bitreversed(j, self.n) for j in range(N)]
            #self.polarcode_mask = pcfun.rm_build_mask(N, K, dSNR) if construct=="rm" else pcfun.RAN87_build_mask(N, K, dSNR) if  construct=="ran87" else pcfun.build_mask(N, K, dSNR)
            self.rprofile = rprofile
            self.polarcode_mask = self.rprofile.build_mask(construct) #in bit-reversal order
            self.rate_profile = self.polarcode_mask[self.bitrev_indices] #in decoding order
            self.frozen_bits = (self.polarcode_mask + 1) % 2  #in bitrevesal order
            self.critical_set_flag = self.rprofile.critical_set_flag((self.polarcode_mask + 1) % 2)
            self.critical_set = pcfun.generate_critical_set((self.polarcode_mask + 1) % 2)
            self.LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            self.stem_LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.stem_BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            #self.list_size = L
            #self.curr_list_size = 1
            self.exp_step = 0
            self.corr_path_exist = 1

            self.dLLR_thresh = 3
            self.last_seen = []
            self.shift_locs = []
            self.PM_last = 0
            self.Loc_last = 0
            self.repeat = False
            self.window_shifted = False
            self.shift_set = []
            self.shft_idx = 0
            self.shft_pmr = []
            #self.model = load_model('model_y_10Kerr_n9_R05_L2_6in_batch418_seq2_binary_epoch30.h5')
            self.cs_seg_cnt = []
            self.seg_tot = 1
            self.flip_cnt = 0
            self.flips_const = 5
            self.bit_idx_B_updating = 0
           #list([iterbale]) is the list constructor
            self.modu = 'BPSK'
            
            self.ml_exploring_mu_min_idx = 0
            self.ml_last_mu_max = 0

            self.A = pcfun.A(self.polarcode_mask, N, K)
            self.pe = np.zeros(N, dtype=float) #pcfun.pe_dega(self.codeword_length,
                                                   #self.information_size,
                                                   #self.designSNR)
            self.sigma = 0
            self.snrb_snr = 'SNRb'
            self.Delta = 0
            self.T = 0
            self.iter_clocks = 0
            self.total_clocks = 0
            self.max_clocks = 5000
            self.total_steps = 0
            self.total_additions = 0
            self.total_comparisons = 0
            self.iter = 0
            self.err_init = 0
            self.prnt_proc = 0
            #Collecting statistics:
            self.bit_err_cnt = np.zeros(N, dtype=int)
            self.tot_err_freq = np.zeros(10, dtype=int)
            

    def __repr__(self):
        return repr((self.codeword_length, self.information_size, self.designSNR))
#__str__ (read as "dunder (double-underscore) string") and __repr__ (read as "dunder-repper" (for "representation")) are both special methods that return strings based on the state of the object.

    def mul_matrix(self, precoded):
        """multiplies message of length N with generator matrix G"""
        """Multiplication is based on factor graph"""
        N = self.codeword_length
        polarcoded = precoded
        for i in range(self.n):
            if i == 0:
                polarcoded[0:N:2] = (polarcoded[0:N:2] + polarcoded[1:N:2]) % 2
            elif i == (self.n - 1):
                polarcoded[0:int(N/2)] = (polarcoded[0:int(N/2)] + polarcoded[int(N/2):N]) % 2
            else:
                enc_step = int(pcfun.np.power(2, i))
                for j in range(enc_step):
                    polarcoded[j:N:(2 * enc_step)] = (polarcoded[j:N:(2 * enc_step)]
                                                    + polarcoded[j + pcfun.np.power(2, i):N:(2 * enc_step)]) % 2
        return polarcoded
    # --------------- ENCODING -----------------------

    def profiling(self, info):
        """Apply polar code mask to information message and return profiled message"""
        profiled = pcfun.np.zeros(self.codeword_length, dtype=int) #array
        profiled[self.polarcode_mask == 1] = info
        self.trdata = copy.deepcopy(profiled)
        return profiled

    def precode(self, info):
        """Apply polar code mask to information message and return precoded message"""
        precoded = pcfun.np.zeros(self.codeword_length, dtype=int) #array
        precoded[self.polarcode_mask == 1] = info
        self.trdata = copy.deepcopy(precoded)
        return precoded
    

    def encode(self, info, issystematic: bool):
        """Encoding function"""
        # Non-systematic encoding
        encoded = self.precode(info)
        #encoded = self.precode_seg(info)
        if self.prnt_proc==1:
            print("[ ", end='')
            for i in range(self.codeword_length):
                print(encoded[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        #print(encoded)
        if not issystematic:
            polarcoded = self.mul_matrix(encoded)
        # Systematic encoding based on non-systematic encoding
        else:
            polarcoded = self.mul_matrix(encoded)
            polarcoded *= self.polarcode_mask
            polarcoded = self.mul_matrix(polarcoded)
            # ns_encoded = self.mul_matrix(self.precode(info))
            # s_encoded = [self.polarcode_mask[i] * ns_encoded[i] for i in range(self.codeword_length)]
            # return self.mul_matrix(s_encoded)
        return polarcoded



    def pac_encode(self, info, conv_gen, mem):
        """Encoding function"""
        # Non-systematic encoding
        if self.prnt_proc==1:
            print("A=[ ", end='')
            for i in range(self.codeword_length):
                print(self.polarcode_mask[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        V = self.precode(info)
        if self.prnt_proc==1:
            print("V=[ ", end='')
            for i in range(self.codeword_length):
                print(V[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        U = pcfun.conv_encode(V, conv_gen, mem)
        if self.prnt_proc==1:
            print("U=[ ", end='')
            for i in range(self.codeword_length):
                print(U[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        X = self.mul_matrix(U)
        return X