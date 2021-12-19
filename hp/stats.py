'''
Created on Dec. 19, 2021

@author: cefect

commons for statistica analysis
    usually these are just templates
'''
import itertools
import pandas as pd
import numpy as np

import scipy.stats
import sklearn
import statsmodels.api as sm


from hp.plot import Plotr

class SimpleRegression(Plotr):
    
    
    def regression(self, #get regressions from different modules and add to plot
                  df_raw,
                     xcoln='area',
                       ycoln='gvimp_gf',
                       figsize=None,
                  ):
        
        
        log = self.logger.getChild('regression')
        
 
        plt = self.plt
        
        #======================================================================
        # figure setup
        #======================================================================
        ax = self.get_ax(title='LinearRegression %s vs %s'%(xcoln, ycoln), figsize=figsize)
        
        #=======================================================================
        # add data
        #=======================================================================
        """
        plt.show()
        """
        ax.set_ylabel(ycoln)
        ax.set_xlabel(xcoln)
        
        ax.scatter(df_raw[xcoln], df_raw[ycoln], color='black', marker='.', s=10)
        
        markers = itertools.cycle((',', '+', 'o', '*'))

        
        #=======================================================================
        # get each regression
        #=======================================================================
        res_d = dict()
        for i, (aName, func, plotkwargs, xloc) in enumerate((
            #('scipy',lambda:self.scipy_lineregres(df_raw, xcoln=xcoln, ycoln=ycoln),{'color':'green'}, 0.1),
            #('sk',lambda:self.sk_linregres(df_raw, xcoln=xcoln, ycoln=ycoln),{'color':'red'}, 0.5),
            ('sm',lambda:self.sm_linregres(df_raw, xcoln=xcoln, ycoln=ycoln),{'color':'blue'}, 0.7),
            )):
            log.info('%i %s'%(i, aName))
            #===================================================================
            # #build the model
            #===================================================================
            
            pfunc, res_d[aName] = func()
            
            
            #=======================================================================
            # plot
            #=======================================================================
            """
            plt.show()
            """
            
            
            ax.plot(df_raw[xcoln], pfunc(df_raw[xcoln]), label=aName, 
                    #marker=next(markers), markersize=2,
                    linewidth=0.2, **plotkwargs)
            
            # annotate

            self.add_anno(res_d[aName], aName, ax=ax, xloc=xloc)
        
 
            
            log.info('finished on %s w/ \n    %s'%(aName, res_d[aName]))
            
        
        #=======================================================================
        # wrap
        #=======================================================================
        ax.legend()
        ax.grid()
        
        self.output_fig(ax.figure, logger=log, fmt='svg')
        

    
    
    def sns_linRegres(self, df_raw,
                       xcoln='area',
                       ycoln='gvimp_gf',
                       hue_col = 'use1_gf',
                       ):
        """just getting pretty plots from seaborn"""
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('sns_linRegres')
        sns = self.sns
        plt = self.plt
        
        
        #=======================================================================
        # loop each plot
        #=======================================================================
        res_d = dict()
        for i, (aName, func) in enumerate({
            'lmplot1':lambda:sns.lmplot(x=xcoln, y=ycoln, data=df_raw, markers='+'),
            'lm_use':lambda:sns.lmplot(x=xcoln, y=ycoln, data=df_raw, hue=hue_col, markers='+'),
            }.items()):
        
            #get the figure
            facetgrid = func()
            
            fig = facetgrid.figure
            fig.suptitle(aName)
        
            res_d[aName] = self.output_fig(fig, logger=log, fmt='svg')
            
        return res_d
        

    def scipy_lineregres(self,
               df_raw,
 
                       xcoln='area',
                       ycoln='gvimp_gf',
 
 
 
               ):
        
        #=======================================================================
        # defaults
        #=======================================================================
 
        log = self.logger.getChild('scipy_lineregres')
        
        #=======================================================================
        # setup data
        #=======================================================================
        df1 = df_raw.loc[:, [xcoln, ycoln]].dropna(how='any', axis=0)
        xtrain, ytrain=df1[xcoln].values, df1[ycoln].values
        #=======================================================================
        # scipy linregress--------
        #=======================================================================
        lm = scipy.stats.linregress(xtrain, ytrain)
        
        predict = lambda x:np.array([lm.slope*xi + lm.intercept for xi in x])
        
        
        return predict, {'rvalue':lm.rvalue, 'slope':lm.slope, 'intercept':lm.intercept}
        
        
        

        

        
    def sk_linregres(self,
               df_raw,
                       xcoln='area',
                       ycoln='gvimp_gf',
 
               ):
        """
        need another package to get confidence intervals
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('sk_linregres')
 
 
        #=======================================================================
        # data prep
        #=======================================================================
        df1 = df_raw.loc[:, [xcoln, ycoln]].dropna(how='any', axis=0)
        
 
        
        xtrain, ytrain = df1[xcoln].values.reshape(-1,1), df1[ycoln].values.reshape(-1,1)
        
        #=======================================================================
        # scikit learn--------
        #=======================================================================
        #=======================================================================
        # train model
        #=======================================================================
        regr = sklearn.linear_model.LinearRegression(positive=True, fit_intercept=False)
        
        # Train the model using the training sets
        estimator = regr.fit(xtrain, ytrain)
        
 
        
        #=======================================================================
        # metrics
        #=======================================================================
        rvalue = regr.score(xtrain, ytrain)
        #print(regr.get_params())
        

        
        """
        help(type(regr))
        help(linear_model.LinearRegression)
        help(regr.fit)
        help(regr)
        pfunc(df1[xcoln])
        """
        pfunc = lambda x:regr.predict(x.values.reshape(-1,1))
        #=======================================================================
        # annotate
        #=======================================================================
        return pfunc, {'rvalue':rvalue, 'slope':regr.coef_[0], 'intercept':regr.intercept_}
        
 
    
    def sm_linregres(self,
                df_raw,
                intercept=False,
                       xcoln='area',
                       ycoln='gvimp_gf',
 
 
 
               ):
        
        log = self.logger.getChild('sm_linregres')
        
        #=======================================================================
        # setup data
        #=======================================================================
        df1 = df_raw.loc[:, [xcoln, ycoln]].dropna(how='any', axis=0)
        xtrain, ytrain=df1[xcoln].values, df1[ycoln].values
        
        
        #=======================================================================
        # OLS model
        #=======================================================================
        if intercept:
            X = sm.add_constant(xtrain)
        else:
            X = xtrain
        
        model = sm.OLS(ytrain, X)
        lm = model.fit()
        print(lm.summary())
        
        if intercept:
            intercept, slope = lm.params[0], lm.params[1]
            pfunc = lambda x:lm.predict(sm.add_constant(x))
        else:
            pfunc = lambda x:lm.predict(x)
            intercept, slope = 0, lm.params[0]
        #lm.predict(X)
        """
        pfunc(xtrain)
        """
        
        return pfunc, {'rvalue':lm.rsquared, 'slope':slope, 'intercept':intercept}
        
    
    def add_anno(self,
            res_d,label, ax=None,
            xloc=0.1, yloc=0.8,
            ):

 
        astr = '\n'.join(['%s = %.2f'%(k,v) for k,v in res_d.items()])
        
        anno_obj = ax.text(xloc, yloc,
                           '%s \n%s'%(label, astr),
                           transform=ax.transAxes, va='center')
        
        
        return astr
        
    def get_ax(self,
               figsize=None,
               title='LinearRegression',
               ):
        
        
        if figsize is None: figsize=self.figsize
        
        plt = self.plt
        
        plt.close()
        fig = plt.figure(0,
                         figsize=figsize,
                     tight_layout=False,
                     constrained_layout = False,
                     #markertype='.',
                     )
        
        fig.suptitle(title)
        return fig.add_subplot(111)
        
        
 