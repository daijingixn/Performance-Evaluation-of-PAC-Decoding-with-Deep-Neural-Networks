from operator import itemgetter
#itemgetter(item) return a callable object that fetches item from its operand using the operandâ€™s __getitem__() method. If multiple items are specified, returns a tuple of lookup values
import numpy as np
import math
from scipy.stats import norm
import csv

class rateprofile:
    #The __init__() function is called automatically 
    #every time the class is being used to create a new object.
    #he self parameter is a reference to the current instance of the class, 
    #and is used to access variables that belong to the class.
    #It does not have to be named self, 
    #but it has to be the first parameter of any function in the class.
    def __init__(self, N, Kp, dSNR):
        self.N = N
        self.n = int(math.log2(N))
        self.Kp = Kp #K plus redundancy :nonfrozen_bits
        self.dsnr_db = dSNR
        self.profile = []


    def bitreversed(self, num: int, n) -> int:
        """"""
        return int(''.join(reversed(bin(num)[2:].zfill(n))), 2)
    
    def bhattacharyya_param(self):
        # bhattacharya_param = [0.0 for i in range(N)]
        bhattacharya = np.zeros(self.N, dtype=float)
        # snr = pow(10, design_snr / 10)
        snr = np.power(10, self.dsnr_db / 10)
        bhattacharya[0] = np.exp(-snr)
        for level in range(1, int(np.log2(self.N)) + 1):
            B = np.power(2, level)
            for j in range(int(B / 2)):
                T = bhattacharya[j]
                bhattacharya[j] = 2 * T - np.power(T, 2)
                bhattacharya[int(B / 2 + j)] = np.power(T, 2)
        return bhattacharya
    
    
    def phi_inv(self, x: float):
        if (x>12):
            return 0.9861 * x - 2.3152
        elif (x<=12 and x>3.5):
            return x*(0.009005 * x + 0.7694) - 0.9507
        elif (x<=3.5 and x>1):
            return x*(0.062883*x + 0.3678)- 0.1627
        else:
            return x*(0.2202*x + 0.06448)
     #Mean-LLR obtained from Density Evolution by Gaussian Approximation (DEGA) method    
    def mllr_dega(self):
        mllr = np.zeros(self.N, dtype=float)
        # snr = pow(10, design_snr / 10)
        #dsnr = np.power(10, dsnr_db / 10)
        sigma_sq = 1/(2*self.Kp/self.N*np.power(10,self.dsnr_db/10))
        mllr[0] = 2/sigma_sq
        #mllr[0] = 4 * K/N * dsnr
        for level in range(1, int(np.log2(self.N)) + 1):
            B = np.power(2, level)
            for j in range(int(B / 2)):
                T = mllr[j]
                mllr[j] = self.phi_inv(T)
                mllr[int(B / 2 + j)] = 2 * T
        #mean = 2/np.square(sigma)
        #var = 4/np.square(sigma)
        return mllr
    
    def pe_dega(self):
        mllr = self.mllr_dega()
        pe = np.zeros(self.N, dtype=float)
        for ii in range(self.N):
            #z = (mllr - mean)/np.sqrt(var)
            #pe[ii] = 1/(np.exp(mllr[ii])+1)
            #pe[ii] = 1 - norm.cdf( np.sqrt(mllr[ii]/2) )
            pe[ii] = 0.5 - 0.5 * math.erf( np.sqrt(mllr[ii])/2 )
        return pe
    
    def A(self, mask):
        j = 0
        A_set = np.zeros(self.Kp, dtype=int)
        for ii in range(self.N):
            if mask[ii] == 1:
                A_set[j] = self.bitreversed(ii, self.n)
                j += 1
        A_set = np.sort(A_set)
        return A_set
    
    def polarization_weight(self):
        w = np.zeros(self.N, dtype=float)
        n = int(np.log2(self.N))
        for i in range(self.N):
            wi = 0
            binary = bin(i)[2:].zfill(n)
            for j in range(n):
                wi += int(binary[j])*pow(2,(j*0.25))
            w[i] = wi
        return w
    
    
    def countOnes(self, num:int):
        ones = 0
        binary = bin(num)[2:]
        len_bin = len(binary)
        for i in range(len_bin):
            if binary[i]=='1':
                ones += 1
        return(ones)
    
    def row_wt(self):
        w = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            w[i] = self.countOnes(i)
        return w

    def min_row_wt(self):
        w = self.row_wt()
        min_w = self.n
        for i in range(self.N):
            if self.profile[i] == 1 and w[i] < min_w:
                min_w = w[i]
        return min_w
        
    def rows_wt(self,wt):
        w = self.row_wt()
        rows = []
        for i in range(self.N):
            if self.profile[i] == 1 and w[i] == wt:
                rows.append(self.bitreversed(i, self.n))
        return rows

    def rows_wt_flag(self,wt):
        w = self.row_wt()
        rows = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            if self.profile[i] == 1 and w[i] == wt:
                rows[i] = 1
        return rows
    
    def bh_build_mask(self):
        """Generates mask (frozen/info bit indicator vector)
        in mask 0 means frozen bit, 1 means information bit"""
        # each bit has 3 attributes
        # [order, bhattacharyya_param/mllr, frozen (0)/ imformation (1) flag for the position]
        mask = [[i, 0.0, 1] for i in range(self.N)]
        # Build mask using Bhattacharya values
        #values = row_wt(N, K)
        #reliability = self.mllr_dega()
        reliability = self.bhattacharyya_param()
        # set bhattacharyya values
        for i in range(self.N):
            mask[i][1] = reliability[i]
        # sort channels due to bhattacharyya values
        #mask = sorted(mask, key=itemgetter(1), reverse=False)   #DEGA, RM
        mask = sorted(mask, key=itemgetter(1), reverse=True)    #bhattacharyya
        # set mask[i][2] in 1 for channels with K lowest bhattacharyya values
        for i in range(self.N-self.Kp):
            mask[i][2] = 0
        # sort channels with respect to index
        mask = sorted(mask, key=itemgetter(0))
        # return non-frozen flag vector
        return np.array([i[2] for i in mask])


    def dega_build_mask(self):
        """Generates mask (frozen/info bit indicator vector)
        in mask 0 means frozen bit, 1 means information bit"""
        # each bit has 3 attributes
        # [order, bhattacharyya_param/mllr, frozen (0)/ imformation (1) flag for the position]
        mask = [[i, 0.0, 1] for i in range(self.N)]
        # Build mask using Bhattacharya values
        #values = row_wt(N, K)
        reliability = self.mllr_dega()
        #reliability = bhattacharyya_param()
        # set bhattacharyya values
        for i in range(self.N):
            mask[i][1] = reliability[i]
        # sort channels due to bhattacharyya values
        mask = sorted(mask, key=itemgetter(1), reverse=False)   #DEGA, RM
        #mask = sorted(mask, key=itemgetter(1), reverse=True)    #bhattacharyya
        # set mask[i][2] in 1 for channels with K lowest bhattacharyya values
        for i in range(self.N-self.Kp):
            mask[i][2] = 0
        # sort channels with respect to index
        mask = sorted(mask, key=itemgetter(0))
        # return non-frozen flag vector
        return np.array([i[2] for i in mask])

    def pw_build_mask(self):
        """Generates mask (frozen/info bit indicator vector)
        in mask 0 means frozen bit, 1 means information bit"""
        # each bit has 3 attributes
        # [order, bhattacharyya_param/mllr, frozen (0)/ imformation (1) flag for the position]
        mask = [[i, 0.0, 1, 0] for i in range(self.N)]
        # Build mask using Bhattacharya values
        #values = row_wt(N, K)
        for i in range(self.N):
            mask[i][3] = self.bitreversed(i,self.n)

        reliability = self.polarization_weight()
        #reliability = bhattacharyya_param()
        # set bhattacharyya values
        for i in range(self.N):
            mask[i][1] = reliability[i]
        # sort channels due to bhattacharyya values
        mask = sorted(mask, key=itemgetter(1), reverse=False)   #DEGA, RM
        #mask = sorted(mask, key=itemgetter(1), reverse=True)    #bhattacharyya
        # set mask[i][2] in 1 for channels with K lowest bhattacharyya values
        for i in range(self.N-self.Kp):
            mask[i][2] = 0
        # sort channels with respect to index
        mask = sorted(mask, key=itemgetter(0))
        #mask_rev = sorted(mask, key=itemgetter(3))
        #mask[self.bitreversed(27,self.n)][2] = 0
        #mask[self.bitreversed(81,self.n)][2] = 1
        # return non-frozen flag vector
        return np.array([i[2] for i in mask])


    #def critical_set_flag(self,frozen_bits:np.int):
    def critical_set_flag(self,frozen_bits:int):
        #cnt = 0
        #critical_set = np.zeros(N, dtype=np.int8)
        hw = []
        critical_set = []
        #A = -1 * np.ones((self.n + 1, self.N), dtype=np.int)    #an extra row for frozen_bits
        A = -1 * np.ones((self.n + 1, self.N), dtype=int)    #an extra row for frozen_bits
        for ii in range(self.N):
            A[-1, self.bitreversed(ii,self.n)] = frozen_bits[ii]
        #A[-1, :] = frozen_bits
        for i in range(self.n-1,-1,-1):
            for j in range(0,np.power(2,(i))):
                A[i, j] = A[i + 1, 2 * j] + A[i + 1, 2 * j + 1]
    
        for i in range(0,self.n+1): #levels
            for j in range(0,np.power(2,(i))): #nodes in levels
                if A[i, j] == 0:
                    index_1 = j
                    index_2 = j
                    for k in range(1, self.n - i+1): #expansion to lower levels
                        index_1 = 2 * index_1 #first bit of rate-1 sub-block in each level
                        index_2 = 2 * index_2 + 1
                        for p in range(index_1, index_2+1):
                            A[i + k, p] = -1 #to avoid considering those nodes again in lower levels
                    critical_set.append(index_1)
                    #critical_set[cnt] = index_1    #first bit in rate-1 node.
                    #cnt += 1
                    """
                    hw0 = (bin(index_1)[2:].zfill(self.n)).count('1')
                    hw.append(hw0)
        hw_min = min(hw)
        #print(len(hw))
        cs_len = len(critical_set)
        idx = 0
        while idx < cs_len:
            hw1 = (bin(critical_set[idx])[2:].zfill(self.n)).count('1')
            #print(hw1)
            if hw1 > hw_min:
                #print(idx)
                cs_len -= 1
                critical_set.pop(idx)
                hw.pop(idx)
            else:
                idx += 1
        """       
        critical_set.sort() #reverse = True
        critical_set_flag = np.zeros(self.N, dtype=int)
        k = 0
        for i in range(self.N):
            if critical_set[k] == i:
                critical_set_flag[i] = 1
                k += 1
                if len(critical_set) == k:
                    break
        return critical_set_flag
        #return np.array(critical_set)
        #critical_set = np.sort(critical_set[critical_set != 0])
    
    
    
    def rmPolar_build_mask(self):
        """Generates mask of polar code
        in mask 0 means frozen bit, 1 means information bit"""
        # each bit has 3 attributes
        # [order, bhattacharyya value, frozen / imformation position]
        # 0 - frozen, 1 - information
        mask = [[i, 0, 0.0, 1] for i in range(self.N)]
        # Build mask using Bhattacharya values
        wt = self.row_wt() # row_wt(i)=2**(wt(bin(i)), value=wt(bin(i))
        mllr = self.mllr_dega()
        #values = bhattacharyya_param(N, design_snr)
        #Bit Error Prob.
        # set bhattacharyya values
        for i in range(self.N):
            mask[i][1] = wt[i]
            mask[i][2] = mllr[i]
        # Sort the channels by Bhattacharyya values
        weightCount = np.zeros(self.n+1, dtype=int)
        for i in range(self.N):
            weightCount[wt[i]] += 1
        bitCnt = 0
        k = 0
        while bitCnt + weightCount[k] <= self.N-self.Kp:
            for i in range(self.N):
                if wt[i]==k:
                    mask[i][3] = 0
                    bitCnt += 1
            k += 1
        mask2 = []
        for i in range(self.N):
            if mask[i][1] == k:
                mask2.append(mask[i])
        mask2 = sorted(mask2, key=itemgetter(2), reverse=True)   #DEGA 原本为False
        remainder = (self.N-self.Kp)-bitCnt
        available = weightCount[k]
        for i in range(remainder):
            mask[mask2[i][0]][3] = 0
        # non-frozen flag vector
        rate_profile = np.array([i[3] for i in mask])
        #mask = sorted(mask, key=itemgetter(0))  #sort based on bit-index
        # return positions bits
        #Modify the profile:
        """
        toFreeze = [21]
        toUnfreeze = [18]
        n = int(math.log2(N))
        for i in range(len(toFreeze)):
            #rate_profile[bitreversed(toFreeze[i], n)] = 0
            #rate_profile[bitreversed(toUnfreeze[i], n)] = 1
            rate_profile[toFreeze[i]] = 0
            rate_profile[toUnfreeze[i]] = 1
        """    
        
        return rate_profile
    


    def build_mask(self, profile):
        if profile == "bh":
            self.profile = self.bh_build_mask()
        elif profile == "dega":
            self.profile = self.dega_build_mask()
        elif profile == "rm-polar":
            self.profile = self.rmPolar_build_mask()
        elif profile == "pw":
            self.profile = self.pw_build_mask()
        return self.profile

