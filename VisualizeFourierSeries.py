#!/usr/bin/env python
# coding: utf-8

### Import libraries
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integral

class FourierSeries():
    '''
    FourierSeries class is a Fourier series object. It computes arbitrarily
    large number of Fourier coefficients, plot out the grapahs, and computes
    the absolute as well as Lp norm of the differences.
    '''

    def __init__(self):
        self.function = None
        self.numTerms = 0
        self.coef = None

    def set_func(self,f):
        self.function = f

    def set_numTerms(self,N):
        self.numTerms = N

    def _vectorized_coef_computer(self):
        pi = math.pi

        def compute_coef(n):
            numerical_int = integral.quad(lambda x: self.function(x)*np.sin(n*x),
            0, pi)
            return 2/pi * numerical_int[0]

        return np.vectorize(compute_coef)


    def get_coefficient(self,n):
        '''
        @param
            n: (int) the n-th Fourier coefficient
        @return
            the n-th Fourier coefficient
        '''

        pi = math.pi
        numerical_int = integral.quad(lambda x: self.function(x)*np.sin(n*x),
        0, pi)

        return 2/pi * numerical_int[0]

    def get_coef_array(self):
        '''
        @return
            (np.array) the first self.numTerms Fourier coefficients as an array
        '''

        vectorized = self._vectorized_coef_computer()
        return vectorized(np.arange(self.numTerms+1))

    def _compute_coef_array(self):
        self.coef = self.get_coef_array()

    def compute_fourier_series(self, term, gap):
        '''
        @param
            term: (array) the terms that need to be plotted
            gap: (float) delta_x for plots
        @return
            Fourier series
        '''

        if self.coef is None:
            self._compute_coef_array()

        x_coord = np.arange(0,3.15,gap)
        M = len(x_coord)

        terms = np.zeros(shape=(self.numTerms, M), dtype=float)

        def _compute_term(x,coef,n):
            return coef*np.sin(n*x)
        _compute_term_vectorized = np.vectorize(_compute_term)
        for n in range(self.numTerms):
            terms[n] = _compute_term_vectorized(x_coord,self.coef[n], n)

        return terms


    def plot_fourier_series(self, term, gap):
        x_coord = np.arange(0,3.15,gap)
        M = len(x_coord)
        terms = self.compute_fourier_series(term, gap)
        partial_sum = np.cumsum(terms, axis=0)
        actual = np.zeros(M)

        for m in range(M):
            actual[m] = self.function(x_coord[m])

        fig = plt.figure(figsize=(8,8))
        for t in term:
            plt.plot(x_coord, partial_sum[t],label=(str(t)+' terms'))
        plt.plot(x_coord, actual, 'b--', label='actual')
        plt.legend(loc="upper right")
        plt.axis('equal')
        plt.savefig('fourier.png')
