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
import sklearn.linear_model
import statsmodels.api as sm


from hp.plot import Plotr, view


def get_confusionMat(yt, preds, classes_ar):
            
    return pd.DataFrame(sklearn.metrics.confusion_matrix(yt, preds), 
                 columns=['pred_'+e for e in classes_ar],
                 index=classes_ar)

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
        
 
    

    def sm_prepData(self, df_raw, ycoln, intercept):
        #=======================================================================
    # setup data
    #=======================================================================
        df_train = df_raw.dropna(subset=[ycoln], how='any', axis=0)
        xtrain = df_train.drop(ycoln, axis=1)
        ytrain = df_train[ycoln]
    #=======================================================================
    # OLS model
    #=======================================================================
        if intercept:
            X = sm.add_constant(xtrain)
        else:
            X = xtrain
        return ytrain, X

    def sm_linregres(self,
                df_raw,
                ycoln,
                intercept=True,

 

               ):
        """
        this seems to be the most powerfule module of those explored
        provides confidence intervals of the model coefficients
        and of the observations
        documentation is bad
        """
        
        log = self.logger.getChild('sm_linregres')
        
        ytrain, X = self.sm_prepData(df_raw, ycoln, intercept)
        
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
    
    
    def sm_MNLogit(self,
                df_raw,
                ycoln,
                intercept=False, #not sure what it means to not have an intercept
                logger=None):
        """
        couldn't get this to work
        """
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('sm_MNLogit')
        
        #=======================================================================
        # setup data
        #=======================================================================
        yt_raw, X = self.sm_prepData(df_raw, ycoln, intercept)
        
 #==============================================================================
 #        #dummie test
 #        def dums(ser):
 #            x=ser[0]
 # 
 #            if x < X.quantile(0.33)[0]:
 #                return 'low'
 #            if x > X.quantile(0.66)[0]:
 #                return 'hi'
 #            else:
 #                return 'med'
 #        
 #        ytrain = X.apply(dums, axis=1).rename('size').to_frame()
 #==============================================================================
        
        #=======================================================================
        # #switch to integers
        # int_d = dict(zip(yt_raw.unique(), range(len(yt_raw.unique()))))
        # 
        # ytrain = yt_raw.replace(int_d)
        #=======================================================================
        
        ytrain=yt_raw
 
        
        model = sm.MNLogit(ytrain, X, missing='raise')
        model.data.ynames
         
         
        #fit with data
        regRes = model.fit()
        log.info(regRes.summary())
        
        if intercept:
            raise IOError('dome')
            res_d = {'intercept':regRes.params[0],
             'slope':regRes.params[1],
             'slope_conf':regRes.conf_int()[1],
             'inter_conf':regRes.conf_int()[0],
             }
 
            
            pfunc = lambda x:regRes.predict(sm.add_constant(x))
        else:
            #regRes.predict(X).sum(axis=1)
            regRes.predict(X).eq(regRes.predict(X).max(axis=1), axis=0).sum()
            model.endog_names
            regRes.pred_table()
            regRes.cov_params()
            regRes.summary()
            regRes.conf_int()

            preds = regRes.predict(X)
            
            
            
            #identify the column with the maximum value
            np.asarray(preds).argmax(1)
            
            miss_rat = float(regRes.resid_misclassified.sum())/len(ytrain)
            
            pfunc = lambda x:regRes.predict(x)
 
            res_d = {
                'intercept':0,
                 'slope':regRes.params[0],
                 'slope_conf':regRes.conf_int()[0],
                 'inter_conf':np.array([0,0]),
                 }
        

    def get_logit_dummies(self, X):
                #dummie test
        def dums(ser):
            x=ser[0]
 
            if x < X.quantile(0.33)[0]:
                return 'low'
            if x > X.quantile(0.66)[0]:
                return 'hi'
            else:
                return 'med'
            
        ytrain = X.apply(dums, axis=1).rename('size').to_frame()
        return ytrain


    def sk_prepData(self, df_raw, ycoln):
        df_train = df_raw.dropna(subset=[ycoln], how='any', axis=0)

        return df_train.drop(ycoln, axis=1).values, df_train[ycoln].values.reshape(1,-1)[0]

    def sk_MNLogit(self, #multinomal logistic regression with sklearn
                df_raw,
                ycoln,
                intercept=True, #not sure what it means to not have an intercept
                logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('sk_MNLogit')
 
        #=======================================================================
        # setup data
        #=======================================================================
        
        xtrain, ytrain = self.sk_prepData(df_raw, ycoln)
        """
        self.sns.pairplot(df_raw, hue=ycoln)
        """
        
        
        log.info('LogisticRegression on \'%s\' w/ %s'%(
            ycoln, str(xtrain.shape)))
        
        """throwing warnings but still returning predictions?"""
        clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial',
                                                solver ='newton-cg',
                                                max_iter=1000,
                                                ).fit(xtrain,ytrain)
                                                
 
        
        
        clf.get_params()
        
        
        plt = self.plt
        #confusion matrix on training data
        conf_df = get_confusionMat(ytrain, clf.predict(xtrain), clf.classes_)
        
        # Plot the decision boundary. For that, we will assign a color to each
        
        #retrieve the color map
        cmap = plt.cm.get_cmap(name='Paired')
 
                    
        d1 = dict(zip(np.unique(ytrain), range(len(np.unique(ytrain)))))
        
        colors_d = {k:cmap(np.linspace(0, 1, len(d1))[i]) for k,i in d1.items()}
        
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        X = df_raw.drop(ycoln, axis=1)
        x_min, x_max = xtrain[:, 0].min() - 0.5, xtrain[:, 0].max() + 0.5
        y_min, y_max = xtrain[:, 1].min() - 0.5, xtrain[:, 1].max() + 0.5
        #h = 10  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 1000))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        self.plt.figure(1, figsize=(6,6))
        
        raise Error('stopped here... need to remap colors on to prediction')
        pd.DataFrame(Z).replace(colors_d)
        
        plt.pcolormesh(xx, yy, Z, cmap=cmap)
        
        plt.scatter(xtrain[:, 0], xtrain[:, 1], 
                    #c=ytrain, edgecolors="k", cmap=plt.cm.Paired,
                    )
 
 
        
        if intercept:
 
            res_d = {
             'score':clf.score(xtrain, ytrain),
             'confusion':conf_df,
             }
 
            
            pfunc = lambda x:clf.predict(x)
        else:
            raise IOError('not implemented')
 
        return res_d, pfunc
        
    
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
        
        
 