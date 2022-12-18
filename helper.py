import streamlit as st
import plotly.graph_objects as go

colors = ['#b4546c', '#fdc4a2']

dark_primary = '#9a0d31'
light_primary = '#b4546c'
light_primary_2 = '#bc5c6c'
secondary_color = '#fdc4a2'
color_2 = '#f8f4f0'
dark_grey = '#bbb8b5'

group_a = ["Netherlands", "Senegal", "Ecuador",  "Qatar"]

group_b = ["England", "United States", "Iran", "Wales"]

group_c = ["Argentina", "Poland", "Mexico", "Saudi Arabia"]

group_d = ["France", "Australia", "Tunisia", "Denmark"]

group_e = ["Japan", "Spain", "Germany", "Costa Rica"]

group_f = ["Morocco", "Croatia", "Belgium", "Canada"]

group_g = ["Brazil", "Switzerland", "Cameroon", "Serbia"]

group_h = ["Portugal", "Korea Republic", "Uruguay", "Ghana"]

countries_list = group_a + group_b + group_c + group_d + group_e + group_f + group_g + group_h

def get_text(x, unit=''):
    text = f'<p style="font-family:Rubik; color:#b4546c; font-size: 30px;">{x} {unit}</p>'
    return text

def get_centered_text(x, unit=''):
    text = f'<p style="font-family:Rubik; color:#b4546c; font-size: 20px; text-align: center;">{x} {unit}</p>'
    return text

def get_centered_text_2(x, unit=''):
    text = f'<p style="font-family:Rubik; color:#fdc4a2; font-size: 20px; text-align: center;">{x} {unit}</p>'
    return text

def getStringText(x):
    text = f'<p style="font-family:Rubik; color:#f8f4f0; font-size: 20.5px; text-align: center;">{x}</p>'
    return text

def getColoredStringText(x, color):
    text = f'<p style="font-family:Rubik; color:{color}; font-size: 30.5px; text-align: center;">{x}</p>'
    return text

def getHeading(x):
    text = f'<h1 style="font-family:Rubik; color:#fdc4a2; font-size: 60px; text-align: center;">{x}</p>'
    return text



features_list = ['Height', 'Weight', 'PaceTotal', 'DribblingTotal',
             'ShootingTotal', 'DefendingTotal', 'PassingTotal', 'PhysicalityTotal',
             'Crossing', 'Finishing', 'HeadingAccuracy',
             'Curve', 'FKAccuracy', 'BallControl',
             'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
             'LongShots', 'Aggression', 'Positioning', 'Vision', 'Penalties',
             'Composure']

default_features_list = ['PaceTotal', 'DribblingTotal', 'ShootingTotal', 'PassingTotal', 'PhysicalityTotal']


def getData(df):
    player_image = df['PhotoUrl'].unique()[0]
    height = df['Height'].unique()[0]
    weight = df['Weight'].unique()[0]
    preferred_foot = df['PreferredFoot'].unique()[0]

    attacking_work_rate = df['AttackingWorkRate'].unique()[0]
    defensive_work_rate = df['DefensiveWorkRate'].unique()[0]

    # player_value = df['Value(in Euro)'].unique()[0]

    national_team = df['Nationality'].unique()[0]
    club = df['Club'].unique()[0]
    age = df['Age'].unique()[0]
    position = df['BestPosition'].unique()[0]

    return player_image, national_team, club, age, height, weight, position, preferred_foot, attacking_work_rate, defensive_work_rate

def getNationalTeamData(df):
    avg_age = df['Age'].unique()[0]
    avg_height = df['Height'].unique()[0]
    avg_weight = df['Weight'].unique()[0]
    avg_pace = df['PaceTotal'].unique()[0]
    avg_dribbling = df['DribblingTotal'].unique()[0]
    avg_shooting = df['ShootingTotal'].unique()[0]
    avg_defending = df['DefendingTotal'].unique()[0]
    avg_passing = df['PassingTotal'].unique()[0]
    avg_physical = df['PhysicalityTotal'].unique()[0]

    return avg_age, avg_height, avg_weight, avg_pace, avg_dribbling, avg_shooting, avg_defending, avg_passing, avg_physical



def plotScatterPolar(categories, list_1, list_2):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list_1,
        theta=categories,
        fill='toself',
        name='',
        marker_color='#b4546c'
    ))
    fig.add_trace(go.Scatterpolar(
        r=list_2,
        theta=categories,
        fill='toself',
        name='',
        marker_color='#fdc4a2'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,

                range=[0, 100]
            )),
        template='plotly_dark'
    )
    return fig



def plotBarCharts(categories, list_1, list_2):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories,
                         y=list_1,
                         text=list_1,
                         name='Player 1',
                         marker_color=light_primary
                         ))
    fig.add_trace(go.Bar(x=categories,
                         y=list_2,
                         text=list_2,
                         name='Player 2',
                         marker_color=secondary_color
                         ))

    fig.update_layout(
        title='',
        title_font=dict(size=25, color='#f8f4f0', family="Rubik"),
        xaxis_tickfont_size=14,
        xaxis=dict(
            titlefont_size=20,
            tickfont_size=14,
        ),
        barmode='group',
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1  # gap between bars of the same location coordinate.
    )

    fig.update_traces(marker_line_color='#000000',
                      marker_line_width=1.5)

    return fig


cluster_col_list = ['Name', 'Age', 'Height', 'Weight', 'Overall', 'Potential', 'Growth',
       'TotalStats', 'BaseStats', 'ValueEUR', 'WageEUR', 'ReleaseClause',
       'IntReputation', 'WeakFoot', 'SkillMoves', 'PaceTotal', 'ShootingTotal',
       'PassingTotal', 'DribblingTotal', 'DefendingTotal', 'PhysicalityTotal',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
       'STRating',
       'LWRating', 'LFRating', 'CFRating', 'RFRating', 'RWRating', 'CAMRating',
       'LMRating', 'CMRating', 'RMRating', 'LWBRating', 'CDMRating',
       'RWBRating', 'LBRating', 'CBRating', 'RBRating']

team_columns_list = ["Age", "Height", "Weight",
                     "PaceTotal", "DribblingTotal",
                     "ShootingTotal", "DefendingTotal",
                     "PassingTotal", "PhysicalityTotal"]