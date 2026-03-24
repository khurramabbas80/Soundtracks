!pip install "altair<6"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load your data
df = pd.read_csv('/work/pipeline/4.7.Albums_analytics_set.csv')

# 2. Filter for positive values (Log cannot handle 0 or negative numbers)
# This matches the logic used for your scatter plot
filtered_df = df[(df['film_revenue'] > 0) & (df['lfm_album_playcount'] > 0)].copy()

# 3. Setup a 1x2 grid for comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- REVENUE ANALYSIS ---
# Raw Revenue: Use axes[0] instead of axes[0, 0]
stats.probplot(filtered_df['film_revenue'], dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot: Raw Film Revenue\n(Note extreme curve at the end)')

# Log Revenue: Use axes[1] instead of axes[0, 1]
stats.probplot(np.log10(filtered_df['film_revenue']), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Log10 Film Revenue\n(Straight line = Log-Normal Distribution)')

plt.tight_layout()
plt.show()

# Third-party imports
import altair as alt
import pandas as pd

# 1. Load data
df = pd.read_csv('/work/pipeline/4.7.Albums_analytics_set.csv')

# 2. Define award categories (keeping your logic)
award_definitions = {
    'Oscars': {
        'nominees': ['oscar_score_nominee', 'oscar_score_winner', 'oscar_song_nominee', 'oscar_song_winner'],
        'winners': ['oscar_score_winner', 'oscar_song_winner']
    },
    'Golden Globes': {
        'nominees': ['globes_score_nominee', 'globes_score_winner', 'globes_song_nominee', 'globes_song_winner'],
        'winners': ['globes_score_winner', 'globes_song_winner']
    },
    'Critics Choice': {
        'nominees': ['critics_score_nominee', 'critics_score_winner', 'critics_song_nominee', 'critics_song_winner'],
        'winners': ['critics_score_winner', 'critics_song_winner']
    },
    'BAFTA': {
        'nominees': ['bafta_score_nominee', 'bafta_score_winner'],
        'winners': ['bafta_score_winner']
    }
}

# 3. Process status
status_cols = []
for award, cols in award_definitions.items():
    status_col = f'{award}_Status'
    status_cols.append(status_col)
    any_nominee = df[cols['nominees']].fillna(False).any(axis=1)
    any_winner = df[cols['winners']].fillna(False).any(axis=1)
    df[status_col] = 'Not Nominated'
    df.loc[any_nominee, status_col] = 'Nominated'
    df.loc[any_winner, status_col] = 'Winner'

# 4. Filter for valid data
# We specifically ensure revenue is at least 10k to match our visual truncation
financial_df = df[(df['film_budget'] > 0) & 
                  (df['film_revenue'] >= 10000) & 
                  (df['lfm_album_playcount'] > 0)].copy()

# 5. Transform to Long Format
long_df = financial_df.melt(
    id_vars=['film_title', 'film_revenue', 'film_popularity', 'lfm_album_playcount'],
    value_vars=status_cols,
    var_name='Award_Type',
    value_name='Award_Status'
)
long_df['Award_Type'] = long_df['Award_Type'].str.replace('_Status', '')

status_order = ['Not Nominated', 'Nominated', 'Winner']
long_df['Award_Status'] = pd.Categorical(long_df['Award_Status'], categories=status_order, ordered=True)
long_df = long_df.sort_values('Award_Status')

# ---------------------------------------------------------
# CALCULATION: Medians (Based on the filtered 10k+ dataset)
# ---------------------------------------------------------
med_pop = long_df['film_popularity'].median()
med_rev = long_df['film_revenue'].median()

# ---------------------------------------------------------
# CHART DEFINITION
# ---------------------------------------------------------

# Median Lines
rule_x = alt.Chart(pd.DataFrame({'x': [med_pop]})).mark_rule(
    strokeDash=[4,4], color='#555555', opacity=0.7, strokeWidth=1.5
).encode(x='x:Q')

rule_y = alt.Chart(pd.DataFrame({'y': [med_rev]})).mark_rule(
    strokeDash=[4,4], color='#555555', opacity=0.7, strokeWidth=1.5
).encode(y='y:Q')

# Points
points = alt.Chart().mark_circle(stroke='white', strokeWidth=0.5).encode(
    x=alt.X('film_popularity:Q', 
            title='Film Popularity (Log Scale)', 
            scale=alt.Scale(type='log'), 
            axis=alt.Axis(grid=False, ticks=False, labels=False, domain=False)), 
    
    y=alt.Y('film_revenue:Q', 
            title='Revenue ($) (Log Scale)', 
            # TRUNCATION: Set the domain to start at 10,000
            scale=alt.Scale(type='log', domain=[10000, long_df['film_revenue'].max()], clamp=True), 
            axis=alt.Axis(grid=False, ticks=False, domain=False)),
            
    size=alt.Size('lfm_album_playcount:Q', scale=alt.Scale(range=[10,1500])),
    color=alt.Color('Award_Status:N', scale=alt.Scale(domain=['Winner', 'Nominated', 'Not Nominated'], range=['#7922cc', '#ce0000', '#BDC3C7'])),
    opacity=alt.Opacity('Award_Status:N', scale=alt.Scale(range=[0.8, 0.5, 0.15]), legend=None),
    tooltip=[
        alt.Tooltip('film_title:N', title='Movie'),
        alt.Tooltip('Award_Status:N', title='Status'), # Added this for you
        alt.Tooltip('film_revenue:Q', title='Revenue', format='$,.0f'),
        alt.Tooltip('lfm_album_playcount:Q', title='Album Plays', format=','),
        alt.Tooltip('film_popularity:Q', title='Popularity Score', format='.1f')
    ]
)

layered_chart = alt.layer(
    rule_x, rule_y, points,
    data=long_df
).properties(
    width=320,
    height=250
)

revenue_chart = layered_chart.facet(
    facet='Award_Type:N', 
    columns=2, 
    title='Revenue vs Film & Album Engagement (2015 to 2025)'
).configure_view(
    stroke=None
).resolve_scale(
    x='shared', 
    y='shared'
).interactive()

revenue_chart.display()
