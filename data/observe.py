import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv("19860105-20210731_race_result.csv")
print(df.info())

fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=temp['viewCount'], nbinsx=100))
fig3.update_xaxes(title_text='viewCounts')
fig3.update_yaxes(title_text='Freqency (log)', type='log')
fig3.update_layout(
    title='default',
    height=500, width=700
)
fig3.show()