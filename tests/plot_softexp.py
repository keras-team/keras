from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.activations import softexp
import seaborn as sns
from seaborn import plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
x = [list(np.arange(-5., 6.))]
x = np.matrix(x)
alpha = list(round(a, 1) for a in np.round(np.arange(-1., 1.2, 0.2), 1))
df = pd.DataFrame()
var = K.variable(value=x)
for a in alpha:
    df[str(a)] = K.get_value(softexp(var, a))[0]
df[df > 5] = np.nan
df[df < -5] = np.nan

cdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  1.0, 1.0))}

cdict = {'red': sns.xkcd_rgb["pale red"], 'green': sns.xkcd_rgb['medium green'], 'blue': sns.xkcd_rgb['denim blue']}
cdict
LinearSegmentedColormap('RedGrnBlu', cdict)
cmap = sns.color_palette(cdict.values())
pallette = cmap
sns.set_palette(pallette)
sns.palplot(sns.color_palette(cdict.values(), 10))
df.plot()
plt.show()
rgb = pd.DataFrame(cmap, columns=list('RGB'))
rgb = pd.DataFrame(cmap.values(), columns=list('RGB'))
cmap
cmap.extend
cmap.extend()
colors = [(0.23137254901960785, 0.3568627450980392, 0.5725490196078431),
 (0.2235294117647059, 0.6784313725490196, 0.2823529411764706),
 (0.8509803921568627, 0.32941176470588235, 0.30196078431372547)]
rgb = pd.DataFrame(colors, columns=list('RGB'))
rbg
rgb
alpha
colors = pd.DataFrame(columns=list('RGB'),index=alpha)
colors.iloc[-1.0] = rgb.iloc[0]
colors.iloc[0] = rgb.iloc[0]
colors
colors.iloc[6] = rgb.iloc[1]
colors.iloc[11] = rgb.iloc[2]
colors.iloc[10] = rgb.iloc[2]
colors.iloc[5] = rgb.iloc[1]
colors.iloc[6] = rgb.iloc[1] * pd.np.nan
colors
colors.interpolate()
colors.interpolate(axis=0)
df
colors.interpolate??


colors.iloc[1] = rgb.iloc[0]
colors.iloc[2] = rgb.iloc[0]
colors.iloc[3] = rgb.iloc[0]
colors.iloc[4] = rgb.iloc[1]
colors.iloc[6] = rgb.iloc[1]
colors.iloc[7] = rgb.iloc[2]
colors.iloc[8] = rgb.iloc[2]
colors.iloc[9] = rgb.iloc[2]

pd.concat((colors.iloc[0:3],colors))
colors = pd.concat((colors.iloc[0:3],colors))
colors
colors.rolling(3)
colors.rolling(3).mean()
colors.rolling(4).mean()
pd.concat((colors,colors.iloc[12:15]))
colors = pd.concat((colors,colors.iloc[12:15]))
colors.iloc[7]
colors.iloc[8]
colors.iloc[9]
colors.iloc[10]
colors.iloc[10] = colors.iloc[9]
colors.rolling(5).mean()
colors.rolling(5).mean().iloc[5:]
colors.rolling(5).mean().iloc[5:].len()
len(alpha)
len(colors.rolling(5).mean().iloc[5:])
colors = colors.rolling(5).mean().iloc[5:]
colors.index = alpha

cmap = sns.color_palette(colors.values)
sns.set_palette(cmap)

df.index = x
df
df.index.name = 'x'
df.plot(xlim=[-5, 5], ylim=[-5, 5])
plt.show()
hist
