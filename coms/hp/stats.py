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


from hp.plot import Plotr, view, Error


def get_confusionMat(yt, preds, classes_ar):
            
    return pd.DataFrame(sklearn.metrics.confusion_matrix(yt, preds), 
                 columns=['pred_'+e for e in classes_ar],
                 index=classes_ar)

class SimpleRegression(Plotr):
    
    

    def plot_lm(self,  #plot a single dimension (of a multi-dimensional models)
                df_raw, #complete data (includes training data)
                xcoln, ycoln, res_d, plotkwargs, pfunc, 
                ax=None,
                
                #plot controls
                plot_conf=True,
                plot_lm=True,
                plot_obs=True,
                logger=None):
        #=======================================================================
        # defaults
        #=======================================================================
        plt=self.plt
        if ax is None:
            ax = plt.gca()
        
        if logger is None: logger=self.logger
        log=logger.getChild('plot_lm.%s.%s'%(xcoln, ycoln))
        #=======================================================================
        # data
        #=======================================================================
        df1= df_raw.copy().dropna(subset=[xcoln], how='any').sort_values(xcoln).reset_index(drop=True) #ignore nulls on indep
        df2 = df1.drop(ycoln, axis=1)
        try:
            yser = pfunc(df2)
        except Exception as e:
            raise Error("failed to predict \'%s\' values w/ \n    %s"%(ycoln, e))
        
        #=======================================================================
        # plot model
        #=======================================================================
        if plot_lm:
            ax.plot(df1[xcoln], yser, label=ycoln, 
                #marker=next(markers), markersize=2,
                linewidth=0.5, **plotkwargs)
        """
        plt.show()
        """
        #=======================================================================
        # #confidence
        #=======================================================================
        if 'conf_df' in res_d and plot_conf:
            #get confidence for this dimension
            conf_df = res_d['conf_df'].copy().loc[['min', 'max'], ['const', xcoln]]
            
 
 
            
            #get y values from each bound
            bounds_d = dict()
            for j, (lab, ser) in enumerate(conf_df.iterrows()):
 
                bounds_d[j] = [ser[xcoln] * xi + ser['const'] for xi in ax.get_xlim()]
            
            ax.fill_between(
                ax.get_xlim(), bounds_d[0], bounds_d[1], 
                interpolate=True, 
                hatch=None, color=plotkwargs['color'], alpha=.3) #create a hatch between the points 'y' and the line 'depth'
            
        #=======================================================================
        # observations
        #=======================================================================
        if 'regRes' in res_d and plot_obs:
            regRes = res_d['regRes'] 
            try:
                pred_ols = regRes.get_prediction(sm.add_constant(df2))
            except:
                pred_ols = regRes.get_prediction(df2)
 
            psmry_df = pred_ols.summary_frame()
            
            """
            view(psmry_df.sort_values('mean'))
            """
 
            for label in ["obs_ci_lower", 'obs_ci_upper']:
 
                ax.plot(df1[xcoln].sort_values(), psmry_df[label].sort_values(), label='%s.%s' % (ycoln, label), 
                #marker=next(markers), markersize=2,
                    linewidth=0.4, linestyle='dashed',  **plotkwargs)
        
        #=======================================================================
        # reset limits
        #=======================================================================
        ax.set_ylim(yser.min(), yser.max())
        log.info('finished')
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
        
 
    

 
    
    def sm_transformX(self,
                      xdf,
                       intercept=False,
                      categoricals=[], #list of columns that are categorical
                      ):
        
        #=======================================================================
        # expand categorical data
        #=======================================================================
        if len(categoricals)>0:
            miss_l = set(categoricals).difference(xdf.columns)
            assert len(miss_l)==0, 'requested cols not in data: %s'%miss_l
            
 
            xdf1 = pd.get_dummies(xdf, columns=categoricals)
            
            """
            xtrain[categoricals[0]].value_counts()
            """
        else:
            xdf1 = xdf
            
        
        #=======================================================================
        # OLS model
        #=======================================================================
        if intercept:
            xdf2 = sm.add_constant(xdf1)
        else:
            xdf2 = xdf1
            
        return xdf2

    def sm_linregres(self,
                df_raw,
                ycoln,
                intercept=True, #whether to predict an intercept (or assume y0=0)
                categoricals=[],
                logger=None,

               ):
        """
        this seems to be the most powerfule module of those explored
        provides confidence intervals of the model coefficients
        and of the observations
        documentation is bad
        """
        if logger is None: logger=self.logger
        log = logger.getChild('sm_linregres')
        
        #=======================================================================
        # prep the data
        #=======================================================================
        df_train = df_raw.dropna(how='any', axis=0)
        
        ytrain = df_train[ycoln]        
 
        
        xtrain = self.sm_transformX(df_train.drop(ycoln, axis=1), intercept=intercept, categoricals=categoricals)
        """
        view(xtrain)
        """
        
        #=======================================================================
        # #init the model
        #=======================================================================
        model = sm.OLS(ytrain, xtrain)
        
        #fit with data
        regRes = model.fit()
        log.debug(regRes.summary())
        
        conf_df = pd.DataFrame(regRes.conf_int().values, 
                               columns=['min', 'max'], index=regRes.conf_int().index)
        
        #add predictions
        conf_df = conf_df.join(regRes.params.rename('val')).loc[:, ['val', 'min', 'max']].T
        
        
        #=======================================================================
        # buidl results
        #=======================================================================
        if intercept:
            """???"""
            if not 'const' in regRes.params:
                raise Error('missing param?')
            res_d = {
             'intercept':regRes.params['const'],
             'slope':regRes.params.drop('const'),
 
             }
 
            
            
            #pfunc = lambda x:regRes.predict(x)
        else:
            #pfunc = lambda x:regRes.predict(x)
            
            conf_df['const'] = 0 #add dummy
            
            res_d = {
                'intercept':0.0,
                 'slope':regRes.params,

                 }
            
        res_d.update({
         'conf_df':conf_df, 
         'indep_colns':df_raw.drop(ycoln, axis=1).columns.tolist(),
         'rvalue':regRes.rsquared, 'stderr':regRes.bse[0],'regRes':regRes,'type':'cont'
            })
        
 
        pfunc = lambda xdf, logger=None:self.sm_pfunc(xdf, regRes.predict, intercept=intercept, categoricals=categoricals,
                                         ecolns = xtrain.columns.tolist(), logger=logger)
        #=======================================================================
        # test predictor function 
        #=======================================================================
        try:
            pfunc(df_raw.drop(ycoln, axis=1))
        except Exception as e:
            raise Error('predictor function for \'%s\' fails w/ \n    %s'%(
                ycoln, e))
 
        return pfunc, res_d
    

    def adjust_transform(self, #check column expecattions 
                        
                        xdf,ecolns,  log):
        """
        using categorical indepednet variables in models
        can result in missing values on either the prediction or the training set
        
        """
        assert isinstance(xdf, pd.DataFrame)
        
        #=======================================================================
        # #xvalues for prediction are too sparse
        #=======================================================================
        miss_l = set(ecolns).difference(xdf.columns)
        xdf1 = xdf.copy()
        if len(miss_l) > 0:
            log.warning('insufficent columsn provided (%i)... adding zeros' % len(miss_l))
            xdf1.loc[:, list(miss_l)] = 0
            
            
        #=======================================================================
        # #xvalues for training are too sparse
        #=======================================================================
        miss_l = set(xdf1.columns).difference(ecolns)
 
        if len(miss_l) > 0:
            log.warning('requested preiction with %i columns without training... dropping these' % len(miss_l))
            xdf1 = xdf1.drop(miss_l, axis=1)
            
        #=======================================================================
        # #final check
        #=======================================================================
        miss_l = set(xdf1.columns).symmetric_difference(ecolns)
        assert len(miss_l) == 0
        return xdf1

    def sm_pfunc(self,
                xdf, 
                 rpfunc, #regRes.predict
                 intercept=True, #not sure what it means to not have an intercept
                categoricals=[],
                ecolns=None, #column expectations
                logger=None,
                ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('sm_pfunc')
        #=======================================================================
        # precheck
        #=======================================================================
        assert isinstance(xdf, pd.DataFrame), 'expects a dataframe'

        
        
        xdf1 = self.sm_transformX(xdf, intercept=intercept, categoricals=categoricals)
        
        #=======================================================================
        # column expectations on transformation
        #=======================================================================
        if not ecolns is None:
            xdf1 = self.adjust_transform(xdf1, ecolns, log)
            
        assert len(xdf1)==len(xdf)
        
        rdf = rpfunc(xdf1)
        
        assert np.array_equal(rdf.index, xdf.index)
        
        return rdf
    
    
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


    def plot_classification(self, 
                             xtrain, ytrain, clf,
                             resolution=20, #resolution of decision space
                            cmap=None,
                            ax=None,figsize=(6, 6)
                            ):
        """
    
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html#sphx-glr-auto-examples-linear-model-plot-iris-logistic-py
        
        """
        plt, matplotlib = self.plt, self.matplotlib
        
        #=======================================================================
        # setup plot
        #=======================================================================
        if ax is None:
            plt.close()
            fig = plt.figure(1, 
                        figsize=figsize, 
                        tight_layout=False, 
                        constrained_layout=True)
            
            ax = fig.subplot(111)
            
 

        if cmap is None: cmap = plt.cm.get_cmap(name='Paired')
        #=======================================================================
        # #get colors
        #=======================================================================
            
 
        colors_d = dict(zip(np.unique(ytrain), np.linspace(0, 1, len(np.unique(ytrain)))))
        #=======================================================================
        # decision boundary (w/ mesh)
        #=======================================================================
        # point in the mesh [x_min, x_max]x[y_min, y_max].
 
        x_min, x_max = xtrain[:, 0].min() - 0.5, xtrain[:, 0].max() + 0.5
        y_min, y_max = xtrain[:, 1].min() - 0.5, xtrain[:, 1].max() + 0.5
        
        #h = 10  # step size in the mesh
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=resolution), np.linspace(y_min, y_max, num=resolution))
        
        raise Error('quit here... not sure what to do for more than 2 indep vars')
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        #convert to colors
        """a bit slow"""
        zcolor_ar = pd.DataFrame(Z).replace(colors_d).astype(float).values
        #plot the mesh
        ax.pcolormesh(xx, yy, zcolor_ar, cmap=cmap, shading='auto')
        #=======================================================================
        # observatinos
        #=======================================================================
        
        #zcolor_ar = pd.DataFrame(ytrain).replace(colors_d).astype(float).values
        
        #merge data back together
        df = pd.concat([pd.DataFrame(xtrain, columns=['x', 'y']), pd.Series(ytrain, name='z')], axis=1)
        
        #plot each
        for label, dfi in df.groupby('z'):
 
            ax.scatter(dfi['x'].values,dfi['y'].values,
                       color=cmap(colors_d[label]) , 
                       label=label, 
                       edgecolors='black',linewidths=0.2,alpha=0.8,
                       #cmap=cmap,
                       s=10)
        """
        plt.show()
        """
        ax.legend()
        return ax
        
    def sk_transformX(self,
            xdf, 
            intercept=True,
            categoricals=[],
            ):
        
        assert isinstance(xdf, pd.DataFrame)
        
        if not intercept:
            raise Error('not implemented')
        
 
        #=======================================================================
        # expand categorical data
        #=======================================================================
        if len(categoricals)>0:
            miss_l = set(categoricals).difference(xdf.columns)
            assert len(miss_l)==0, 'requested cols not in data: %s'%miss_l
            
 
            xdf1 = pd.get_dummies(xdf, columns=categoricals)
            
            """
            xtrain[categoricals[0]].value_counts()
            """
        else:
            xdf1 = xdf
            
            
        #=======================================================================
        # wrap
        #=======================================================================
        return xdf1
            
            
        
        
            

    def sk_MNLogit(self, #multinomal logistic regression with sklearn
                df_raw,
                ycoln,
                intercept=True, #not sure what it means to not have an intercept
                categoricals=[],
                logger=None,
                ax=None,
                ):
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('sk_MNLogit.%s'%ycoln)
 
        #=======================================================================
        # setup data
        #=======================================================================
        df_train = df_raw.dropna(subset=[ycoln], how='any', axis=0)

        ytrain = df_train[ycoln].values.reshape(1,-1)[0]
        
        
        xdf = self.sk_transformX(df_train.drop(ycoln, axis=1),
                                    intercept=intercept, categoricals=categoricals)
                                    
        xtrain = xdf.values
 
        
 
        """
        self.sns.pairplot(df_raw, hue=ycoln)
        """
        
        
        log.info('LogisticRegression on \'%s\' w/ %s'%(
            ycoln, str(xtrain.shape)))
        
        #=======================================================================
        # train model
        #=======================================================================
        """throwing warnings but still returning predictions?"""
        clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial',
                                                solver ='newton-cg',
                                                max_iter=1000,
                                                ).fit(xtrain,ytrain)
                                                
 
        
        
        log.debug('fit w/ \n    %s'%(clf.get_params()))
        
        
        #=======================================================================
        # report
        #=======================================================================
 
        #confusion matrix on training data
        conf_df = get_confusionMat(ytrain, clf.predict(xtrain), clf.classes_)
        
        res_d = {
             'score':clf.score(xtrain, ytrain),
             #
             'params':clf.get_params(),
             }
                
        #=======================================================================
        # # Plot-------
        #=======================================================================
        """only setup for 2 indeps now"""
     #==========================================================================
     #    if len(df_raw.columns)<=3:
     #        self.plot_classification(xtrain, ytrain, clf, ax=ax)
     #        
     #        ax.set_ylabel(df_raw.drop(ycoln, axis=1).columns[1])
     #        ax.set_xlabel(df_raw.drop(ycoln, axis=1).columns[0])
     #        """
     #        self.plt.show()
     #        """
     # 
     #        self.add_anno({k:'%.2f'%v for k,v in res_d.items() if isinstance(v, float)}, ycoln, ax=ax)
     #==========================================================================
        
        #=======================================================================
        # get meta
        #=======================================================================
        
        if intercept:
 
            res_d.update({

             })

        else:
            raise IOError('not implemented')
        
        pfunc = lambda x, logger=None:self.sk_pfunc(
            x, clf, intercept=intercept, categoricals=categoricals, ecolns=xdf.columns.tolist(), logger=logger)
        
        #=======================================================================
        # test function
        #=======================================================================
        try:
            pfunc(df_raw.drop(ycoln, axis=1))
        except Exception as e:
            raise Error('predictor func for %s failed w/ \n    %s'%(
                ycoln, e))
            
        res_d.update({
            'pfunc':pfunc, 
            'confusion':conf_df,
            'type':'cat',
            'indep_colns':df_raw.drop(ycoln, axis=1).columns.tolist(),

             })
             
 
        return  res_d
    
    
    def sk_pfunc(self,
                 df_raw,
                 clf,
                 intercept=True, #not sure what it means to not have an intercept
                categoricals=[],
                ecolns=None,
                logger=None,
                ):
        
        #=======================================================================
        # defaults
        #=======================================================================
        if logger is None: logger=self.logger
        log=logger.getChild('sk_pfunc')
        
        assert isinstance(df_raw, pd.DataFrame), 'expects a dataframe'
        bx = df_raw.isna().any(axis=1)
        
        
        #=======================================================================
        # transform
        #=======================================================================
        xdf = self.sk_transformX(df_raw.loc[~bx, :],
                            intercept=intercept, categoricals=categoricals)
        
        
        #=======================================================================
        # check expectations
        #=======================================================================
        if not ecolns is None:
            
            xar = self.adjust_transform(xdf, ecolns, log).values
        else:
            xar = xdf.values
        
        yar = clf.predict(xar)
        
        rser = pd.Series(yar,index=df_raw[~bx].index)
        
        #add empties back
        if bx.any():
            rser = pd.concat([rser, 
                      pd.Series(index=df_raw[bx].index, dtype=rser.dtype)]
            ).loc[df_raw.index]
        
        return rser
        
    
    def add_anno(self,
            d,label, ax=None,
            xloc=0.1, yloc=0.8,**kwargs
            ):

 
        astr = '\n'.join(['%s = %s'%(k,v) for k,v in d.items()])
        
        anno_obj = ax.text(xloc, yloc,
                           '%s \n%s'%(label, astr),
                           transform=ax.transAxes, va='center', **kwargs)
        
        
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
        
        
 