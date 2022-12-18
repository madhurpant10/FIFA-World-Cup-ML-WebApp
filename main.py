import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


from scipy.stats import poisson
from PIL import Image

from helper import *

img = Image.open('Images/world_cup_img.png')
st.set_page_config(page_title="FIFA World Cup Dashboard", page_icon=img)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


world_cup_df = pd.read_csv('Datasets/fifa_world_cup_matches.csv')

players = pd.read_csv('Datasets/players_fifa23.csv')
players = players[players['BestPosition'] != 'GK']
players_list = players['Name']

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Match Prediction", "National Team Analysis", "Player Analysis", "Player Clustering",
                                  "Recommender System", "FAQs"])

## ------------------------------------------ TAB 1 -----------------------------------------
with tab1:

    st.write(getHeading("World Cup Match Prediction"), unsafe_allow_html=True)
    st.image("Images/world_cup_poster.jpg")

    df_home = world_cup_df[['HomeTeam', 'HomeGoals', 'AwayGoals']]
    df_away = world_cup_df[['AwayTeam', 'HomeGoals', 'AwayGoals']]

    df_home = df_home.rename(columns={'HomeTeam':'Team', 'HomeGoals': 'GoalsScored', 'AwayGoals': 'GoalsConceded'})
    df_away = df_away.rename(columns={'AwayTeam':'Team', 'HomeGoals': 'GoalsConceded', 'AwayGoals': 'GoalsScored'})

    df_team_strength = pd.concat([df_home, df_away], ignore_index=True).groupby(['Team']).mean()


    def predict_points(home, away):
        if home in df_team_strength.index and away in df_team_strength.index:
            # goals_scored * goals_conceded
            lamb_home = df_team_strength.at[home, 'GoalsScored'] * df_team_strength.at[away, 'GoalsConceded']
            lamb_away = df_team_strength.at[away, 'GoalsScored'] * df_team_strength.at[home, 'GoalsConceded']
            prob_home, prob_away, prob_draw = 0, 0, 0
            for x in range(0, 11):  # number of goals home team
                for y in range(0, 11):  # number of goals away team
                    p = poisson.pmf(x, lamb_home) * poisson.pmf(y, lamb_away)
                    if x == y:
                        prob_draw += p
                    elif x > y:
                        prob_home += p
                    else:
                        prob_away += p

            points_home = 3 * prob_home + prob_draw
            points_away = 3 * prob_away + prob_draw
            percent_home = (points_home / (points_home + points_away) * 100).round(2)
            percent_away = (points_away / (points_home + points_away) * 100).round(2)
            return percent_home
        else:
            return (0, 0)


    countries = df_team_strength.index.to_list()

    col1, col2 = st.columns(2)

    with col1:
        my_team = st.selectbox("Select a Country:", countries, index=2)

    with col2:
        opponent = st.selectbox("Select Opponent:", countries, index=28)

    my_team_win_percentage = predict_points(my_team, opponent)

    st.header("")  # For Creating Space

    st.write(getColoredStringText(f"{my_team} has a {my_team_win_percentage}% chance of Winning against {opponent}", "#fdc4a2"), unsafe_allow_html=True)

    ## ------------------------------------------ TAB 2 -----------------------------------------
with tab2:
    st.write(getHeading("National Team Analysis"), unsafe_allow_html=True)

    national_team_df = players[players['Nationality'].isin(countries_list)]

    team_stats_df = national_team_df.groupby(['Nationality'])[team_columns_list].mean().reset_index().rename(
        columns={0: 'count'})
    team_stats_df = team_stats_df.round(decimals=2)

    left, middle, right = st.columns(3)

    with left:
        country1 = st.selectbox("Select Country 1:", countries_list, index=8)
        country1_df = team_stats_df[team_stats_df['Nationality'] == country1]

        avg_age_1, avg_height_1, avg_weight_1, avg_pace_1, avg_dribbling_1, avg_shooting_1, avg_defending_1, avg_passing_1, avg_physical_1 = getNationalTeamData(country1_df)

        st.write(get_centered_text(avg_age_1), unsafe_allow_html=True)
        st.write(get_centered_text(avg_height_1, 'cm'), unsafe_allow_html=True)
        st.write(get_centered_text(avg_weight_1, 'kg'), unsafe_allow_html=True)
        st.write(get_centered_text(avg_pace_1), unsafe_allow_html=True)
        st.write(get_centered_text(avg_dribbling_1), unsafe_allow_html=True)
        st.write(get_centered_text(avg_shooting_1), unsafe_allow_html=True)
        st.write(get_centered_text(avg_defending_1), unsafe_allow_html=True)
        st.write(get_centered_text(avg_passing_1), unsafe_allow_html=True)
        st.write(get_centered_text(avg_physical_1), unsafe_allow_html=True)

    with middle:
        st.write("")
        st.write("")
        st.header("")
        st.write(getStringText("Avg Age"), unsafe_allow_html=True)
        st.write(getStringText("Avg Height"), unsafe_allow_html=True)
        st.write(getStringText("Avg Weight"), unsafe_allow_html=True)
        st.write(getStringText("Avg Pace"), unsafe_allow_html=True)
        st.write(getStringText("Avg Dribbling"), unsafe_allow_html=True)
        st.write(getStringText("Avg Shooting"), unsafe_allow_html=True)
        st.write(getStringText("Avg Defending"), unsafe_allow_html=True)
        st.write(getStringText("Avg Passing"), unsafe_allow_html=True)
        st.write(getStringText("Avg Physical"), unsafe_allow_html=True)


    with right:
        country2 = st.selectbox("Select Country 2:", countries_list, index=12)
        country2_df = team_stats_df[team_stats_df['Nationality'] == country2]

        avg_age_2, avg_height_2, avg_weight_2, avg_pace_2, avg_dribbling_2, avg_shooting_2, avg_defending_2, avg_passing_2, avg_physical_2 = getNationalTeamData(country2_df)

        st.write(get_centered_text_2(avg_age_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_height_2, 'cm'), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_weight_2, 'kg'), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_pace_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_dribbling_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_shooting_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_defending_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_passing_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(avg_physical_2), unsafe_allow_html=True)

    st.header("")
    st.write(getColoredStringText("Nationality Feature Distribution", secondary_color), unsafe_allow_html=True)

    col1, col2 = st.columns([0.4, 0.1])

    with col1:
        feature = st.selectbox("Select Feature", team_columns_list)
    with col2:
        num_of_bins = st.number_input("Number of bins:", min_value=1, max_value=5, value=2)

    col1, col2 = st.columns(2)

    with col1:
        show_hist = st.checkbox("Hist", value=True)
    with col2:
        show_curve = st.checkbox("Curve", value=True)

    temp_1 = players[players['Nationality'] == country1][feature]
    temp_2 = players[players['Nationality'] == country2][feature]

    hist_data = [temp_1, temp_2]
    group_labels = ["Country 1", "Country 2"]

    hist_fig = ff.create_distplot(hist_data, group_labels, bin_size=num_of_bins, show_rug=False, show_hist=show_hist,
                             show_curve=show_curve, colors=['#b4546c', '#fdc4a2'])

    hist_fig.update_layout(title_text=f'{feature} Distribution', title_font=dict(size=25, color='#f8f4f0', family="Rubik"), width=800)

    st.plotly_chart(hist_fig)

    ## ------------------------------------------ TAB 3 -----------------------------------------

with tab3:
    st.write(getHeading("Player Analysis"), unsafe_allow_html=True)
    left, middle, right = st.columns(3)

    with left:
        player1 = st.selectbox("Select Player 1:", players_list)
        player1_df = players[players['Name'] == player1]

        photo_1, national_team_1, club_1, age_1, height_1, weight_1, position_1, preferred_foot_1, attacking_work_rate_1, defensive_work_rate_1 = getData(
            player1_df)

        st.image(photo_1)
        st.write(get_centered_text(national_team_1), unsafe_allow_html=True)
        st.write(get_centered_text(club_1), unsafe_allow_html=True)
        st.write(get_centered_text(age_1), unsafe_allow_html=True)
        st.write(get_centered_text(height_1, 'cm'), unsafe_allow_html=True)
        st.write(get_centered_text(weight_1, 'kg'), unsafe_allow_html=True)
        st.write(get_centered_text(position_1), unsafe_allow_html=True)
        st.write(get_centered_text(preferred_foot_1), unsafe_allow_html=True)
        st.write(get_centered_text(attacking_work_rate_1), unsafe_allow_html=True)
        st.write(get_centered_text(defensive_work_rate_1), unsafe_allow_html=True)

    with middle:
        st.header("")
        st.header("")
        st.header("")
        st.write("")
        st.write(getStringText("Nationality"), unsafe_allow_html=True)
        st.write(getStringText("Club"), unsafe_allow_html=True)
        st.write(getStringText("Age"), unsafe_allow_html=True)
        st.write(getStringText("Height"), unsafe_allow_html=True)
        st.write(getStringText("Weight"), unsafe_allow_html=True)
        st.write(getStringText("Position"), unsafe_allow_html=True)
        st.write(getStringText("Foot"), unsafe_allow_html=True)
        st.write(getStringText("Attacking Work Rate"), unsafe_allow_html=True)
        st.write(getStringText("Defensive Work Rate"), unsafe_allow_html=True)

    with right:
        player2 = st.selectbox("Select Player 2:", players_list, index=6)
        player2_df = players[players['Name'] == player2]

        photo_2, national_team_2, club_2, age_2, height_2, weight_2, position_2, preferred_foot_2, attacking_work_rate_2, defensive_work_rate_2 = getData(
            player2_df)

        st.image(photo_2)
        st.write(get_centered_text_2(national_team_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(club_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(age_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(height_2, 'cm'), unsafe_allow_html=True)
        st.write(get_centered_text_2(weight_2, 'kg'), unsafe_allow_html=True)
        st.write(get_centered_text_2(position_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(preferred_foot_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(attacking_work_rate_2), unsafe_allow_html=True)
        st.write(get_centered_text_2(defensive_work_rate_2), unsafe_allow_html=True)

    st.header("")
    st.write(getColoredStringText("Player Feature Comparison", secondary_color), unsafe_allow_html=True)


    categories = ['PaceTotal', 'DribblingTotal',
                  'ShootingTotal', 'DefendingTotal',
                  'PassingTotal', 'PhysicalityTotal']

    stats_1_list = players[players['Name'] == player1][categories].values.flatten().tolist()
    stats_2_list = players[players['Name'] == player2][categories].values.flatten().tolist()

    col1, col2 = st.columns([0.1, 0.9])

    with col2:
        chart = st.radio("Select Chart:", options=['Bar Chart', 'Polar Chart'], horizontal=True)

        if chart == 'Bar Chart':
            players_bar_chart = plotBarCharts(categories, stats_1_list, stats_2_list)
            st.write(players_bar_chart)

        else:
            players_polar_fig = plotScatterPolar(categories, stats_1_list, stats_2_list)
            st.write(players_polar_fig)

    ## ------------------------------------------ TAB 4 -----------------------------------------
with tab4:

    st.write(getHeading("Player Clustering"), unsafe_allow_html=True)

    cluster_df = players[cluster_col_list]

    num_of_players = st.slider("Number of Players from the Top:", min_value=10, max_value=100, value=50)
    num_of_clusters = st.slider("Number of Clusters:", min_value=2, max_value=10, value=5)

    df_cluster = cluster_df.head(num_of_players)
    names = df_cluster['Name'].to_list()

    clustering_type = st.radio("", options=['2D Clustering', '3D Clustering'], horizontal=True)
    if clustering_type == '2D Clustering':

        df_2d = df_cluster.drop(['Name'], axis=1)

        x_2d = df_2d.values
        scaler = preprocessing.MinMaxScaler()
        x_scaled_2d = scaler.fit_transform(x_2d)
        x_norm_2d = pd.DataFrame(x_scaled_2d)

        pca_2d = PCA(n_components=2)
        df_xy = pd.DataFrame(pca_2d.fit_transform(x_norm_2d))

        kmeans_2d = KMeans(n_clusters=num_of_clusters)

        kmeans_2d = kmeans_2d.fit(df_xy)

        labels = kmeans_2d.cluster_centers_

        clusters = kmeans_2d.labels_.tolist()

        df_xy['cluster'] = clusters
        df_xy['Name'] = names
        df_xy.columns = ['x', 'y', 'cluster', 'name']

        cluster_2d = px.scatter(df_xy, x="x", y="y", color="cluster",
                                hover_data=['name'])

        cluster_2d.update_traces(marker=dict(size=14,
                                             line=dict(width=1,
                                                       color='DarkSlateGrey')),
                                 selector=dict(mode='markers'))

        cluster_2d.update_layout(
            title='2D Clustering',
            title_font=dict(size=25, color='#f8f4f0', family="Rubik"))

        cluster_2d.update_coloraxes(showscale=False)

        st.write(cluster_2d)

    else:
        df_3d = df_cluster.drop(['Name'], axis=1)

        x = df_3d.values
        scaler = preprocessing.MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        x_norm = pd.DataFrame(x_scaled)

        pca = PCA(n_components=3)
        reduced = pd.DataFrame(pca.fit_transform(x_norm))

        kmeans = KMeans(n_clusters=num_of_clusters)

        kmeans = kmeans.fit(reduced)

        labels = kmeans.cluster_centers_

        clusters = kmeans.labels_.tolist()

        reduced['cluster'] = clusters
        reduced['Name'] = names
        reduced.columns = ['x', 'y', 'z', 'cluster', 'name']

        fig = px.scatter_3d(reduced, x="x", y="y", z='z', color="cluster",
                            hover_data=['name'])

        fig.update_traces(marker=dict(size=14,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))

        fig.update_layout(
            scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="black", ),
            yaxis=dict(
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="black"),
            zaxis=dict(
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="black"),
        ),
        title='3D Clustering',
        title_font=dict(size=25, color='#f8f4f0', family="Rubik")
                            )

        fig.update_coloraxes(showscale=False)

        st.write(fig)


with tab5:
    st.write(getHeading("Player Recommender System"), unsafe_allow_html=True)

    players_list = players['Name']
    selected_player = st.selectbox("Select a Player:", players_list)

    selected_features = st.multiselect("Select Features: ", options=features_list, default=default_features_list)

    current_df = players[players['Name'] == selected_player]
    selected_player_id = current_df['ID'].unique()[0]

    # field_cols = ['PaceTotal', 'DribblingTotal', 'ShootingTotal', 'PassingTotal']
    field_cols = selected_features
    tr = players[['ID', 'Name'] + field_cols].dropna()
    id_name = players[['Name', 'ID']].set_index('ID')['Name'].to_dict()

    NUM_RECOM = 10

    X = tr[field_cols].dropna().values
    nbrs = NearestNeighbors(n_neighbors=NUM_RECOM + 1, algorithm='ball_tree').fit(X)
    dist, rank = nbrs.kneighbors(X)

    similar_df = pd.DataFrame(columns=[f'rank_{i}' for i in range(1, NUM_RECOM + 1)],
                              index=tr['ID'].values,
                              data=rank[:, 1:])
    dist_df = pd.DataFrame(columns=[f'rank_{i}' for i in range(1, NUM_RECOM + 1)],
                           index=tr['ID'].values,
                           data=dist[:, 1:])

    for cols in list(similar_df):
        tg_col = similar_df[cols]
        new_value = tr['ID'].iloc[tg_col].tolist()
        similar_df[cols] = new_value


    def similar_player(similar_df, player_id):
        player_id = int(player_id)
        player_name = tr[tr['ID'] == player_id]['Name'].values[0]

        sim_ply = similar_df.loc[player_id]
        display_col = ['Name'] + selected_features
        display_df = pd.DataFrame({'ID': similar_df.loc[player_id]}).merge(tr[['ID'] + display_col], how='left', on='ID')[display_col]

        return display_df
        # display(display_df.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))


    player_recommender_df = similar_player(similar_df, selected_player_id)

    st.markdown('_showing recommendations for_ **{}**'.format(selected_player))
    st.dataframe(player_recommender_df)

    ## ------------------------------------------ TAB 6 -----------------------------------------
with tab6:
    st.write(getHeading("Frequently Asked Questions"), unsafe_allow_html=True)

    ########## Expander 1
    expand_faq1 = st.expander("🏆 How was the Match Prediction Calculated?")
    with expand_faq1:
        st.write('''Every Team's Goals Scored and Goals Conceded are calculated and Poisson Distribution is applied.: 
    * The Poisson distribution is a discrete probability distribution that describes the
         number of events occurring in a fixed time interval or region of opportunity.''', unsafe_allow_html=True)


    ########## Expander 2
    expand_faq2 = st.expander("✏️ How to use the Charts effectively?")
    with expand_faq2:
        st.write(''' The Visualizations are built on plotly. Here's how to make the most out of the plot: 
    * The Charts are Interactive and Dynamic. 
    * You can click on Legends to hide/show.
    * You can save a png image of the plot by clicking on the camera icon above. ''', unsafe_allow_html=True)


    ########## Expander 3
    expand_faq3 = st.expander("✨️ How did you Cluster when there are hundreds of features?")
    with expand_faq3:
        st.write(''' Applied Principal Component Analysis(PCA) which converted 65 features into: 
    * 2 Features (x,y) for the 2 Dimensional K-Means Clustering. 
    * 3 Features (x,y, z) for the 3 Dimensional K-Means Clustering.''', unsafe_allow_html=True)

    ########## Expander 4
    expand_faq4 = st.expander("⚽️ How Does the System Recommend similar Players?")
    with expand_faq4:
        st.write(''' The Recommender System makes use of K-Nearest Neighbors Algorithm: 
    * Based on the Selected features, the system plots every player and returns the nearest neighbors.
    * Goal Keepers are not included in the Dataset.''', unsafe_allow_html=True)

    ########## Expander 5
    expand_faq5 = st.expander("💎 Where can I get all the code for this project ?")
    with expand_faq5:
        st.write('''You can find the complete updated code along with the datasets over here: [Github](https://github.com/madhurp10).''',
                 unsafe_allow_html=True)