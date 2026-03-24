import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import altair as alt

# Basic page setup
st.set_page_config(page_title="Film & Awards Analysis", layout="wide")
st.title("Film and Awards Analysis (2015-2025)")

# 1. Load data from the repository
@st.cache_data
def load_data():
    # Changed from Deepnote path to relative path for GitHub/Streamlit
    return pd.read_csv('4.7.Albums_analytics_set.csv')

df = load_data()

# ---------------------------------------------------------
# SECTION 1: REVENUE ANALYSIS (Matplotlib)
# ---------------------------------------------------------
st.header("1. Revenue Distribution Analysis")

filtered_df = df[(df['film_revenue'] > 0) & (df['lfm_album_playcount'] > 0)].copy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw Revenue
stats.probplot(filtered_df['film_revenue'], dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot: Raw Film Revenue')

# Log Revenue
stats.probplot(np.log10(filtered_df['film_revenue']), dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Log10 Film Revenue')

plt.tight_layout()
st.pyplot(fig) # Use st.pyplot to show Matplotlib figures

# ---------------------------------------------------------
# SECTION 2: AWARDS ANALYSIS (Altair)
# ---------------------------------------------------------
st.header("2. Revenue vs Engagement by Award Type")

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

status_cols = []
for award, cols in award_definitions.items():
    status_col = f'{award}_Status'
    status_cols.append(status_col)
    any_nominee = df[cols['nominees']].fillna(False).any(axis=1)
    any_winner = df[cols['winners']].fillna(False).any(axis=1)
    df[status_col] = 'Not Nominated'
    df.loc[any_nominee, status_col] = 'Nominated'
    df.loc[any_winner, status_col] = 'Winner'

financial_df = df[(df['film_budget'] > 0) & 
                  (df['film_revenue'] >= 10000) & 
                  (df['lfm_album_playcount'] > 0)].copy()

long_df = financial_df.melt(
    id_vars=['film_title', 'film_revenue', 'film_popularity', 'lfm_album_playcount'],
    value_vars=status_cols,
    var_name='Award_Type',
    value_name='Award_Status'
)
long_df['Award_Type'] = long_df['Award_Type'].str.replace('_Status', '')

# Chart medians and logic
med_pop = long_df['film_popularity'].median()
med_rev = long_df['film_revenue'].median()

rule_x = alt.Chart(pd.DataFrame({'x': [med_pop]})).mark_rule(
    strokeDash=[4,4], color='#555555', opacity=0.7
).encode(x='x:Q')

rule_y = alt.Chart(pd.DataFrame({'y': [med_rev]})).mark_rule(
    strokeDash=[4,4], color='#555555', opacity=0.7
).encode(y='y:Q')

points = alt.Chart().mark_circle(stroke='white', strokeWidth=0.5).encode(
    x=alt.X('film_popularity:Q', title='Film Popularity (Log)', scale=alt.Scale(type='log')), 
    y=alt.Y('film_revenue:Q', title='Revenue (Log)', scale=alt.Scale(type='log', domain=[10000, long_df['film_revenue'].max()], clamp=True)),
    size=alt.Size('lfm_album_playcount:Q', scale=alt.Scale(range=[10,1500])),
    color=alt.Color('Award_Status:N', scale=alt.Scale(domain=['Winner', 'Nominated', 'Not Nominated'], range=['#7922cc', '#ce0000', '#BDC3C7'])),
    tooltip=['film_title', 'Award_Status', 'film_revenue', 'lfm_album_playcount']
)

revenue_chart = alt.layer(rule_x, rule_y, points, data=long_df).properties(width=320, height=250).facet(
    facet='Award_Type:N', columns=2
).resolve_scale(x='shared', y='shared').interactive()

# Final render on Streamlit
st.altair_chart(revenue_chart, use_container_width=True)