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


from hp.plot import Plotr, view

class SimpleRegression(Plotr):
    
    

    def plot_lm(self, xser, res_d, aName, plotkwargs, pfunc, 
                ax=None):
        #=======================================================================
        # defaults
        #=======================================================================
        plt=self.plt
        if ax is None:
            ax = plt.gca()
        
        #=======================================================================
        # plot model
        #=======================================================================
        ax.plot(xser, pfunc(xser), label=aName, 
            #marker=next(markers), markersize=2,
            linewidth=0.2, **plotkwargs)
        
        #=======================================================================
        # #confidence
        #=======================================================================
        if 'slope_conf' in res_d:
            #get y values from each bound
            bounds_d = dict()
            for j, (m, b) in enumerate(zip(res_d['slope_conf'], res_d['inter_conf'])):
                bounds_d[j] = [m * xi + b for xi in ax.get_xlim()]
            
            ax.fill_between(
                ax.get_xlim(), bounds_d[0], bounds_d[1], 
                interpolate=True, 
                hatch=None, color=plotkwargs['color'], alpha=.3) #create a hatch between the points 'y' and the line 'depth'
            
        #=======================================================================
        # observations
        #=======================================================================
        if 'regRes' in res_d:
            regRes = res_d['regRes'] 
            try:
                pred_ols = regRes.get_prediction(sm.add_constant(xser))
            except:
                pred_ols = regRes.get_prediction(xser)
 
            psmry_df = pred_ols.summary_frame()
            
            """
            view(psmry_df.sort_values('mean'))
            """
            
            
            for label in ["obs_ci_lower", 'obs_ci_upper']:
 
                ax.plot(xser.sort_values(), psmry_df[label].sort_values(), label='%s.%s' % (aName, label), 
                #marker=next(markers), markersize=2,
                    linewidth=0.4, linestyle='dashed', **plotkwargs)
        
 
        
        return ax

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
        ax.set_xlim(0, df_raw[xcoln].max())
        
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
            # plot model
            #=======================================================================
            ax = self.plot_lm(df_raw, xcoln, ax, res_d, aName, plotkwargs, pfunc)
        
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
                intercept=True,
                       xcoln='area',
                       ycoln='gvimp_gf',

               ):
        """
        this seems to be the most powerfule module of those explored
        provides confidence intervals of the model coefficients
        and of the observations
        documentation is bad
        """
        
        log = self.logger.getChild('sm_linregres')
        
        #=======================================================================
        # setup dataTrue
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
        
        #init the model
        model = sm.OLS(ytrain, X)
        
        #fit with data
        regRes = model.fit()
        log.info(regRes.summary())
        
        if intercept:
            res_d = {'intercept':regRes.params[0],
             'slope':regRes.params[1],
             'slope_conf':regRes.conf_int()[1],
             'inter_conf':regRes.conf_int()[0],
             }
 
            
            pfunc = lambda x:regRes.predict(sm.add_constant(x))
        else:
            pfunc = lambda x:regRes.predict(x)
 
            res_d = {
                'intercept':0,
                 'slope':regRes.params[0],
                 'slope_conf':regRes.conf_int()[0],
                 'inter_conf':np.array([0,0]),
                 }
                        
 
        return pfunc, {'rvalue':regRes.rsquared, 'stderr':regRes.bse[0],'regRes':regRes, **res_d}
        
    
    def add_anno(self,
            res_d,label, ax=None,
            xloc=0.1, yloc=0.8,
            ):

 
        astr = '\n'.join(['%s = %.2f'%(k,v) for k,v in res_d.items() if isinstance(v, float)])
        
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
        
        
 