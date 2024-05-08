#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 00:26:46 2023

@author: lefumaqelepo
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig_ext = '.eps'
project_folder = os.path.join(os.getcwd(), '..')
figures_folder = os.path.join(project_folder, 'Figures')
if os.path.exists(figures_folder):
    pass
else:
    os.mkdir(figures_folder)
data_folder = os.path.join(project_folder, 'Data')

# Load data

# Tariff data
def import_data():
    
    tariff_data = pd.read_excel(os.path.join(data_folder, 'electricity_prices.xlsx'))
    tariff_dfs = []
    for col in list(tariff_data.columns)[1:]:
        df = tariff_data.filter(items=['country', col])
        df = df.rename(columns={col:'tariff'})
        tariff_dfs.append(df)
    tariff_df = pd.concat(tariff_dfs)
    tariff_df = tariff_df.groupby(['country']).mean().reset_index()

    path = os.path.join(data_folder, 'GNI.xls')
    gni_data = pd.read_excel(path, skiprows=3, sheet_name='Data')
    gni_data = gni_data.filter(items=['Country Name', 'Country Code', '2017', '2018', '2019'])
    gni_data = gni_data[gni_data['Country Name'].isin(tariff_df.country.unique())]
    gni_dfs = []
    for col in list(gni_data.columns)[2:]:
        df = gni_data.filter(items=['Country Name', 'Country Code', col])
        df = df.rename(columns={'Country Name':'country', 'Country Code':'code', col:'gni'})
        gni_dfs.append(df)
    gni_df = pd.concat(gni_dfs)
    gni_df = gni_df.groupby(['country', 'code']).mean().reset_index()
    
    tariff_gni_df = pd.merge(gni_df, tariff_df, on=['country'], how='left')
    return tariff_gni_df[['country', 'code', 'gni', 'tariff']]


def import_shape_files():
    shp = gpd.read_file(os.path.join(data_folder, 'afr_g2014_2013_0', 'afr_g2014_2013_0.shp'))
    shp = shp.filter(items=['ADM0_NAME', 'geometry'])
    shp.loc[10, 'ADM0_NAME'] = "Congo, Rep."
    shp.loc[11, 'ADM0_NAME'] = "Cote d'Ivoire"
    shp.loc[12, 'ADM0_NAME'] = "Congo, Dem. Rep."
    shp.loc[41, 'ADM0_NAME'] = "Eswatini"
    shp.loc[44, 'ADM0_NAME'] = "Tanzania"
    shp = shp.rename(columns={'ADM0_NAME':'country'})
    return shp

def import_poverty_data(codes = 'none'):
    poverty_data = pd.read_excel(os.path.join(data_folder, 'API_11_DS2_en_excel_v2_5359207.xls'), skiprows=3)
    cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2010', '2011', '2012',
            '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    poverty_data = poverty_data[cols]
    poverty_data = poverty_data.rename(columns={'Country Name':'country', 'Country Code':'code', 'Indicator Name':'i_name', 'Indicator Code':'i_code'})
    i_codes = ['SI.DST.FRST.20', 'SI.DST.FRST.10', 'SI.DST.50MD', 'SI.DST.10TH.10', 'SI.DST.05TH.20', 'SI.DST.04TH.20',
               'SI.DST.03RD.20', 'SI.DST.02ND.20']
    poverty_data = poverty_data[poverty_data.i_code.isin(i_codes)]
    if type(codes) != str:
        poverty_data = poverty_data[poverty_data.code.isin(codes)]
    X = poverty_data.drop(columns=['country', 'code', 'i_name', 'i_code']).to_numpy()
    pdata = poverty_data[['country', 'code', 'i_name', 'i_code']].copy()
    pdata['pop_share'] = np.nanmean(X, axis=1)
    return pdata

def import_population_data():
    population_data = pd.read_excel(os.path.join(data_folder, 'API_SP.POP.TOTL_DS2_en_excel_v2_5358476.xls'), skiprows=3)
    cols = ['Country Name', 'Country Code', '2017', '2018', '2019']
    population_data = population_data[cols]
    population_data = population_data.rename(columns={'Country Name':'country', 'Country Code':'code'})
    X = population_data.drop(columns=['country', 'code']).to_numpy()
    pdata = population_data[['code']].copy()
    pdata['population'] = X.mean(axis=1)
    return pdata
    
def import_elec_consumption_data():
    consumption_data = pd.read_excel(os.path.join(data_folder, 'API_EG.USE.ELEC.KH.PC_DS2_en_excel_v2_5359521.xls'), skiprows=3)
    consumption_data = consumption_data[['Country Code', '2014']]
    consumption_data = consumption_data.rename(columns={'Country Code':'code', '2014':'consumption'})
    return consumption_data
    
def data_merger(base_df, population_df, consumption_df, geodata_df):
    gdf = geodata_df[geodata_df.country.isin(base_df.country.unique())]
    pdf = population_df[population_df.code.isin(base_df.code.unique())]
    cdf = consumption_df[consumption_df.code.isin(base_df.code.unique())]
    base1 = pd.merge(base_df, cdf, on=['code'], how='left')
    comb_data = gpd.GeoDataFrame(pd.merge(base1, gdf, on=['country'], how='left'))
    comb_data = gpd.GeoDataFrame(pd.merge(comb_data, pdf, on=['code'], how='left'))
    comb_data['standard_spend'] = comb_data.gni*5/100
    comb_data['standard_tariff'] = comb_data.standard_spend*100/365
    comb_data['tariff_overpay'] = comb_data.tariff - comb_data.standard_tariff
    comb_data['ratio'] = (comb_data.standard_tariff - comb_data.tariff)/(comb_data.standard_tariff + comb_data.tariff)
    return comb_data
    
def plot_gni_tariff(df):
            
    fig, ax = plt.subplots(dpi=150)
    ax.set_xscale("log", base=10)
    ax.scatter('gni', 'tariff', s=10, data=df, color='b', alpha=0.7, edgecolors='none', label='Country')
    ax.axhline(df.tariff.mean(), ls='--', color='k')
    ax.axvline(df.gni.mean(), ls=':', color='r')
    
    for (x, y, val) in zip(df.gni, df.tariff, df.code):
        ax.annotate(val, xy=(1.05*x, y-0.5), fontsize=5)
    ax.set_ylabel('Tariff [US cents/kWh]')
    ax.set_xlabel('Gross National Income Per Capita [US $]')
    fig.legend(loc='lower left', bbox_to_anchor=(0.05, 0))
    plt.savefig(os.path.join(figures_folder, 'gni_tariff' + fig_ext), bbox_inches='tight')

def visuals(comb_data):
    df = comb_data.copy()
    #df = df.sort_values(by='tariff_overpay')
    opay = df[df.tariff > df.standard_tariff]
    upay = df[df.tariff < df.standard_tariff]
    
    # Tariffs comparison
    fig, ax = plt.subplots(dpi=150)
    ax.set_yscale('log', base=10)
    ax.scatter('tariff', 'standard_tariff', s=5, color='r', data=opay, label='Tariff Unaffordable Countries: N = {}'.format(len(opay)))
    ax.scatter('tariff', 'standard_tariff', s=5, color='b', data=upay, label='Tariff Affordable Countries: N = {}'.format(len(upay)))
    ax.plot(np.arange(1, 46), np.arange(1, 46), ls='--', color='k')
    #ax.grid(visible=True, which='both')
    for (x, y, val) in zip(df.tariff, df.standard_tariff, df.code):
        ax.annotate(val, xy=(x+0.5, 1.05*y), fontsize=5)
    ax.set_ylabel('Standard Tariff [$]')
    ax.set_xlabel('Tariff [$]')
    fig.legend(fontsize=8, loc=3, ncol=2, bbox_to_anchor=(0.08, -0.07))
    plt.savefig(os.path.join(figures_folder, 'tariff_analysis' + fig_ext), bbox_inches='tight')
    
    fig, ax = plt.subplots(dpi=150)
    dfc = df.copy()
    dfc = df.sort_values(by=['ratio'])
    dfc = dfc[~dfc.code.isin(['ERI', 'SOM', 'SSD'])]
    tf = dfc.tariff.values*100
    stf = dfc.standard_tariff.values*100
    y1 = np.array(list(map(lambda a, b: a*100/b if a < b else 100, tf, stf)))
    y2 = np.array(list(map(lambda a, b: b*100/a if b < a else 100, tf, stf)))
    v = 0.4
    w = np.min(np.vstack([y1, y2]), axis=0)
    y1_t = np.where(y2 > w, w, 0)
    y2_t = np.where(y1 > w, w, 0)
    x = np.arange(0, len(y1))
    xtick_labs = dfc.code.values
    ax.bar(x - v/2, y1, v, fill=False, linewidth=0, hatch='/////', label='Overpay')
    ax.bar(x + v/2, y2, v, fill=False, linewidth=0, hatch='-----', label='Underpay')
    ax.bar(x, w, color='w', alpha=1)
    ax.bar(x + v/2, y1, v, color='b', label='Actual tariff')
    ax.bar(x - v/2, y2, v, color='r', label='Standard tariff')
    #ax.step(x, y1, where='mid', color='b', ls='-', lw=1)
    # ax.step(x, y2, where='mid', color='k', ls='-', lw=1)
    ax.set_xticks(x, xtick_labs, rotation=90, fontsize=7)
    x_major_ticks = ax.xaxis.get_major_ticks()
    x_major_tick_labels = ax.xaxis.get_majorticklabels()
    for i in x:
        if (i + 1) % 2 == 1:
            x_major_ticks[i].tick1line.set_markersize(7)
        elif (i + 1) % 2 == 0:
            x_major_ticks[i].tick1line.set_markersize(24)
            x_major_tick_labels[i].set_y(-0.08)
    ax.set_ylabel('Fraction of maximum [%]')
    ax.set_xlabel('Countries')
    
    hs, ls = ax.get_legend_handles_labels()
    lbls = ['Actual tariff', 'Standard tariff', 'Overpay', 'Underpay']
    hndls = []
    for lbl in lbls:
        hndls.append(hs[ls.index(lbl)])
        
    fig.legend(hndls, lbls, ncol=4, fontsize=8, loc=3, bbox_to_anchor=(0.1, -0.15), frameon=False)
    plt.savefig(os.path.join(figures_folder, 'tariff_compare' + fig_ext), bbox_inches='tight')
    
    # Standard tariff map
    fig, ax = plt.subplots(dpi=150)
    mapper(df, ax, 'brg', 'standard_tariff', 'Standard Tariff [US cents]')
    plt.savefig(os.path.join(figures_folder, 'standard_tariff_map' + fig_ext), bbox_inches='tight')
    
    # Affordability map
    fig, ax = plt.subplots(dpi=150)
    mapper(df, ax, 'cividis', 'ratio', 'Affordability', norm=None)
    df_lower_cs = df[df.ratio < 0].geometry.centroid
    df_upper_cs = df[df.ratio > 0].geometry.centroid
    df_upper_cs.plot(marker='P', color='k', markersize=4, ax=ax, label='Underpaying')
    df_lower_cs.plot(marker='P', color='greenyellow', markersize=4, ax=ax, label='Overpaying')
    fig.legend(loc=8, ncol=1, fontsize=7, frameon=True, fancybox=False, bbox_to_anchor=(0.25, 0.2, 0.1, 0.4), facecolor='darkgrey')
    plt.savefig(os.path.join(figures_folder, 'affordability_map' + fig_ext), bbox_inches='tight')
    
    # Population map
    fig, ax = plt.subplots(dpi=150)
    mapper(comb_data, ax, 'cividis', 'population')
    plt.savefig(os.path.join(figures_folder, 'population' + fig_ext), bbox_inches='tight')
    
    # Consumption plot
    fig, ax = plt.subplots(dpi=150)
    dfc = comb_data.copy()
    dfc = dfc.sort_values(by=['consumption'])
    x = np.arange(0, len(dfc))
    ax.bar(x, dfc.consumption.values, color='b', label='Mean consumption')
    ax.axhline(365, color='k', ls='--', label='Standard consumption')
    ax.set_xticks(x, dfc.code, rotation=90, fontsize=7)
    x_major_ticks = ax.xaxis.get_major_ticks()
    x_major_tick_labels = ax.xaxis.get_majorticklabels()
    for i in x:
        if (i + 1) % 2 == 1:
            x_major_ticks[i].tick1line.set_markersize(7)
        elif (i + 1) % 2 == 0:
            x_major_ticks[i].tick1line.set_markersize(24)
            x_major_tick_labels[i].set_y(-0.08)
    ax.set_xlabel('Countries')
    ax.set_ylabel('Electricity consumption [kWh]')
    fig.legend(loc=7, frameon=False, fontsize='x-small', bbox_to_anchor=(0.9, 0.5))
    plt.savefig(os.path.join(figures_folder, 'consumption' + fig_ext), bbox_inches='tight')
    

def mapper(df, ax, cmap, variable, varname=None, norm='LogNorm'):
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    if varname == None:
        varname = variable
    if norm == 'LogNorm':
        norm = colors.LogNorm(vmin=df[variable].min(), vmax=df[variable].max())
    else:
        norm = colors.Normalize(vmin=df[variable].min(), vmax=df[variable].max())
    df.plot(column=variable, ax=ax, legend=True, cmap=cmap, norm=norm, edgecolors='k', lw=0.3, 
            missing_kwds={'color':'whitesmoke', 'hatch':'////'},
            legend_kwds={'shrink':0.5, 'label':'{}'.format(varname)})
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis("off")
    
    
def plot_population_income_pcts(df):
    data = df.pop_share.values
    Y = data.reshape((len(df)//8, 8))
    x = np.arange(0, len(df)//8)
    fig, ax = plt.subplots(dpi=150)
    ax.bar(x, Y[:, 4], bottom=Y[:, 0] + Y[:, 5] + Y[:, 6] + Y[:, 7],  color='blueviolet', label='second highest 10%')
    ax.bar(x, Y[:, 3], bottom=100 - Y[:, 3], color='blue', label='highest 10%')
    ax.bar(x, Y[:, 5], bottom=Y[:, 0] + Y[:, 7] + Y[:, 6], color='lightsteelblue', label='fourth 20%')
    ax.bar(x, Y[:, 6], bottom=Y[:, 0] + Y[:, 7], color='lightgray', label='third 20%')
    ax.bar(x, Y[:, 7], bottom=Y[:, 0], color='darkgray', label='second 20%')
    ax.bar(x, Y[:, 0], color='dimgray', label='second lowest 10%')
    ax.bar(x, Y[:, 1], color='k', label='lowest 10%')
    ax.set_xticks(x, list(df.code.unique()), rotation=90, fontsize=7)
    x_major_ticks = ax.xaxis.get_major_ticks()
    x_major_tick_labels = ax.xaxis.get_majorticklabels()
    for i in x:
        if (i + 1) % 2 == 1:
            x_major_ticks[i].tick1line.set_markersize(7)
        elif (i + 1) % 2 == 0:
            x_major_ticks[i].tick1line.set_markersize(24)
            x_major_tick_labels[i].set_y(-0.08)
    ax.set_ylabel('Share of income [%]')
    ax.set_xlabel('Countries')
    
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['highest 10%', 'second highest 10%', 'fourth 20%', 'third 20%', 'second 20%', 'second lowest 10%', 'lowest 10%']
    new_handles = []
    for lab in new_labels:
        new_handles.append(handles[labels.index(lab)])
    fig.legend(new_handles, new_labels, ncol=4, fontsize=8, loc=3, frameon=False, bbox_to_anchor=(0.0, -0.18))
    plt.savefig(os.path.join(figures_folder, 'population_income_brackets' + fig_ext), bbox_inches='tight')


def tariff_design(attr_df, poverty_df):
    
    attrs = attr_df.copy()
    attrs = attrs[['code', 'gni', 'tariff', 'standard_tariff', 'consumption', 'population']]
    # attrs['total_income'] = attrs['gni']*attrs['population']
    # attrs['total_consumption'] = attrs['consumption']*attrs['population']
    
    population_groups = [
        'Income share held by lowest 20%',
        'Income share held by second 20%',
        'Income share held by third 20%',
        'Income share held by fourth 20%',
        'Income share held by highest 20%'
        ]
    
    ptitles = ['FRST_20_ishare', 'SEC_20_ishare', 'THR_20_ishare', 'FOR_20_ishare', 'FIF_20_ishare']
    for idx, (pg, pt) in enumerate(zip(population_groups, ptitles)):
        attrs = attrs.merge(poverty_df[poverty_df.i_name == pg][['code', 'pop_share']], on=['code'], how='left')
        attrs = attrs.rename(columns={'pop_share':pt})
        attrs['std_tar_'+str(idx+1)] = 5*attrs['standard_tariff']*attrs[pt]/100
    return attrs

def plot_income_group_tariff(attr_df):
    
    df = attr_df[['code', 'tariff', 'standard_tariff', 'std_tar_1', 'std_tar_2', 'std_tar_3', 'std_tar_4', 'std_tar_5']].copy()
    df = df.dropna()
    X = df.drop(columns=['code', 'tariff']).to_numpy()
    r, c = X.shape
    Xmax = X.max(axis=1).flatten()
    Divisor = np.vstack([Xmax]*c).T
    
    X = X/Divisor*100
    df.loc[:, 'tariff'] = df.tariff/Xmax
    df.loc[:, 'standard_tariff'] = X[:, 0]
    df.loc[:, 'std_tar_1'] = X[:, 1]
    df.loc[:, 'std_tar_2'] = X[:, 2]
    df.loc[:, 'std_tar_3'] = X[:, 3]
    df.loc[:, 'std_tar_4'] = X[:, 4]
    df.loc[:, 'std_tar_5'] = X[:, 5]
    df = df.sort_values(by=['standard_tariff'], ascending=False)
    
    fig, ax = plt.subplots(dpi=150, subplot_kw=dict(polar=True))
    x = np.linspace(0, 2*np.pi, len(df), endpoint=False)
    xtick_labs = df.code.values
    
    X_OFFSET = 0.05 # to control how far the scale is from the plot (axes coordinates)
    def add_scale(ax, ylabel):
        # add extra axes for the scale
        rect = ax.get_position()
        rect = (rect.xmin-X_OFFSET, rect.ymin+rect.height/2, # x, y
                rect.width, rect.height/2) # width, height
        scale_ax = ax.figure.add_axes(rect)
        # hide most elements of the new axes
        for loc in ['right', 'top', 'bottom']:
            scale_ax.spines[loc].set_visible(False)
        scale_ax.tick_params(bottom=False, labelbottom=False)
        scale_ax.patch.set_visible(False) # hide white background
        # adjust the scale
        scale_ax.spines['left'].set_bounds(*ax.get_ylim())
        # scale_ax.spines['left'].set_bounds(0, ax.get_rmax()) # mpl < 2.2.3
        scale_ax.set_yticks(ax.get_yticks())
        scale_ax.set_ylim(ax.get_rorigin(), ax.get_rmax())
        scale_ax.set_ylabel(ylabel, fontsize=8)
        
    # ax.scatter(x, df.tariff.values, s=4, marker='d', color='steelblue')
    # ax.plot(x, df.tariff.values, lw=1, color='steelblue', label=r'Tariff $\Omega$')
    ax.scatter(x, df.standard_tariff.values, s=4, color='magenta')
    ax.plot(x, df.standard_tariff.values, lw=1, color='magenta', ls='--', label='Standard tariff')
    ax.scatter(x, df.std_tar_1.values, s=4, color='darkgray')
    ax.plot(x, df.std_tar_1.values, lw=1, color='darkgray', label='Lowest 20%')
    ax.scatter(x, df.std_tar_2.values, s=5, color='dimgray')
    ax.plot(x, df.std_tar_2.values, lw=1, color='dimgray', label='Second 20%')
    ax.scatter(x, df.std_tar_3.values, s=5, color='black')
    ax.plot(x, df.std_tar_3.values, lw=1, color='black', label='Third 20%')
    ax.scatter(x, df.std_tar_4.values, s=5, color='red')
    ax.plot(x, df.std_tar_4.values, lw=1, color='red', label='Fourth 20%')
    ax.scatter(x, df.std_tar_5.values, s=5, color='blue')
    ax.plot(x, df.std_tar_5.values, color='blue', label='Highest 20%')
    ax.set_theta_zero_location('N')
    ax.set_rorigin(-30)
    ax.set_ylim(0, 100)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks, [])
    ax.set_xticks(x, xtick_labs, fontsize=8)
    add_scale(ax, 'Fraction of maximum [%]')
    
    fig.legend(fontsize=7, ncol=3, frameon=True, loc=8, fancybox=False, bbox_to_anchor=(0.5, -0.07))
    plt.savefig(os.path.join(figures_folder, 'tariff_design' + fig_ext), bbox_inches='tight')
    
    
# def bubble_plot(df, ax, color, annotate=False):
#     max_t_df = df[df.tariff == df.tariff.max()]
#     max_g_df = df[df.gni == df.gni.max()]
#     ax.scatter('gni', 'tariff', s='tariff', data=df, color=color, alpha=0.7, edgecolors='none', label=str(df.year.values[0]))
#     ax.axhline(df.tariff.mean(), ls='--', lw=0.5, color=color)
#     ax.axvline(df.gni.mean(), ls=':', lw=0.5, color=color)
#     if annotate:
#         ax.annotate(max_t_df.country.values[0], xy=(max_t_df.gni + 200, max_t_df.tariff), fontsize=8)
#         ax.annotate(max_g_df.country.values[0], xy=(max_g_df.gni, max_g_df.tariff + 3), fontsize=8)

def main():
    data = import_data()
    shp = import_shape_files()
    population_data = import_population_data()
    consumption_data = import_elec_consumption_data()
    cmb = data_merger(data, population_data, consumption_data, shp)
    
    plot_gni_tariff(data)
    visuals(cmb)
    poverty_data = import_poverty_data(cmb.code.values)
    
    plot_population_income_pcts(poverty_data)
    
    tar_df = tariff_design(cmb, poverty_data)
    plot_income_group_tariff(tar_df)
    
if __name__=='__main__':
    
    main()
    