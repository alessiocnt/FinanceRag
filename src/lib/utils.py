from lib.dataset import *   
from lib.data_handler import *
from lib.embeddings import *
from lib.vector_store import *
from lib.RAG_pipeline import *
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd

def split_column_metrics(df):
    for metric in ['ndcg', 'recall', 'mrr']:
        df[f'{metric}_5'] = df[metric].apply(lambda x: x['@5'])
        df[f'{metric}_10'] = df[metric].apply(lambda x: x['@10'])
        df.drop(metric, axis=1, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'dataset'}, inplace=True)
    return df

def check_table_presence(dataset_manager):
    for DATASET_NAME in dataset_manager.list_datasets():
        corpus, queries, qrels = dataset_manager.load_dataset(DATASET_NAME)
        corpus, queries = reduce_dataset_size(corpus, queries, qrels)
        print(f"Dataset: {DATASET_NAME} has {len(corpus)} documents in corpus")

        multi_table_docs = {'1': 0, '2': 0, '3': 0, '4': 0, '+':0}  # Initialize the keys with 0
        for k,v in corpus.items():
            text = v[0]
            tables = extract_tables(v[0])
            if tables:
                if len(tables)==1:
                    multi_table_docs['1'] += 1
                if len(tables)==2:
                    multi_table_docs['2'] += 1
                if len(tables)==3:
                    multi_table_docs['3'] += 1
                if len(tables)==4:
                    multi_table_docs['4'] += 1
                if len(tables)>4:
                    multi_table_docs['+'] += 1

        # unique print with all the information
        print(f"""\t{multi_table_docs['1']} documents with 1 table
            {multi_table_docs['2']} documents with 2 tables
            {multi_table_docs['3']} documents with 3 tables
            {multi_table_docs['4']} documents with 4 tables 
            {multi_table_docs['+']} documents with more than 4 tables\n""")

def plot_count_words(corpus_df, queries_df):
    # Calculate text lengths based on word count
    corpus_df['word_count'] = corpus_df['text'].apply(lambda x: len(x.split()))
    queries_df['word_count'] = queries_df['text'].apply(lambda x: len(x.split()))

    # Calculate max and average word counts
    max_word_count_corpus = corpus_df['word_count'].max()
    average_word_count_corpus = round(corpus_df['word_count'].mean(), 2)
    max_word_count_queries = queries_df['word_count'].max()
    average_word_count_queries = round(queries_df['word_count'].mean(), 2)

    print(f"Max word count: {max_word_count_corpus}")
    print(f"Average word count: {average_word_count_corpus}")
    print(f"Max word count in queries: {max_word_count_queries}")
    print(f"Average word count in queries: {average_word_count_queries}")

    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Corpus Word Count Distribution", "Queries Word Count Distribution"))
    fig.add_trace(go.Histogram(
        x=corpus_df['word_count'],
        nbinsx=50,
        marker_color='blue',
        name="Corpus Word Count"
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=queries_df['word_count'],
        nbinsx=50,
        marker_color='green',
        name="Queries Word Count"
    ), row=1, col=2)
    fig.update_layout(
        title_text="Word Count Distribution in Corpus and Queries",
        showlegend=False,
        height=400,
        width=800
    )
    fig.update_xaxes(title_text="Word Count")
    fig.update_yaxes(title_text="Frequency")
    fig.show()

def plot_table_results(df_results_ns, df_results_ss, df_results_ls):
    df_ns_long = pd.melt(df_results_ns, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ss_long = pd.melt(df_results_ss, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ls_long = pd.melt(df_results_ls, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ns_long['approach'] = 'NS'
    df_ss_long['approach'] = 'SS'
    df_ls_long['approach'] = 'LS'
    df_long = pd.concat([df_ns_long, df_ss_long, df_ls_long])

    # Extract metric base (ndcg, recall, mrr) and type (@5, @10)
    df_long["metric_base"] = df_long["metric"].str.replace(r'_5|_10', '', regex=True)
    df_long["metric_type"] = df_long["metric"].str.extract(r'(5|10)').astype(int)

    approach_colors = {
        'NS': '#fc6a03',  # Red for NS
        'SS': '#55bebd ',  # Green for SS
        'LS': '#e377c2'   # Blue for LS
    }

    # Define datasets and metrics
    datasets = df_results_ls['dataset'].unique()
    metrics = ['ndcg', 'recall', 'mrr']

    # Create subplots: 3 rows (metrics) × 4 columns (datasets)
    fig = sp.make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{dataset}" for metric in metrics for dataset in datasets],
        shared_yaxes=False
    )
    metric_positions = {'ndcg': 1, 'recall': 2, 'mrr': 3}
    dataset_positions = {dataset: i+1 for i, dataset in enumerate(datasets)}

    for _, row in df_long.iterrows():
        dataset = row['dataset']
        metric = row['metric_base']
        metric_type = row['metric_type']
        approach = row['approach']
        value = row['value']
        
        row_idx = metric_positions[metric]  # Metrics define the row
        col_idx = dataset_positions[dataset]  # Datasets define the column

        if metric_type == 10:
            fig.add_trace(
                go.Bar(
                x=[approach],  
                y=[value], 
                name=f"{metric.upper()} @{metric_type}",
                marker=dict(color=approach_colors[approach]),  # Color by approach
                opacity=0.6,
                textposition='outside',
                text=[value]  # Add the value as text on the bar
                ),
                row=row_idx, col=col_idx
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=[approach],  
                    y=[value], 
                    name=f"{metric.upper()} @{metric_type}",
                    marker=dict(color=approach_colors[approach]),  # Color by approach
                    opacity=0.6,
                    textposition='outside'  # Position text outside the bar
                ),
                row=row_idx, col=col_idx
            )

    # Update layout
    fig.update_layout(
        title="Comparison of Approaches per Dataset and Metric",
        height=1400, width=1400,
        bargap=0.1,  # Increase gap between bars to make them narrower
        barmode="overlay",
        xaxis_title="Approaches",
        yaxis_title="Metric Score",
        showlegend=False
    )
    # Update axis titles per row (corrected for metric names)
    for metric, row in metric_positions.items():
        fig.update_yaxes(title_text=f"{metric.upper()} Score", range=[0, 1], row=row, col=1)
        for col in range(1, len(datasets) + 1): 
            fig.update_xaxes(title_text="Approaches", row=row, col=col)

    for col in range(1, len(datasets) + 1): 
        fig.update_yaxes(range=[0, 0.9], row=1, col=col)
        fig.update_yaxes(range=[0, 1], row=2, col=col)
        fig.update_yaxes(range=[0, 0.8], row=3, col=col)

    fig.update_yaxes(range=[0, 0.6], row=1, col=3)
    fig.update_yaxes(range=[0, 0.6], row=2, col=3)
    fig.update_yaxes(range=[0, 0.6], row=3, col=3)
 
    fig.show()

def plot_one_approach(df_results_ls, approach_name):
    df_ls_long = pd.melt(df_results_ls, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ls_long['approach'] = approach_name

    # Extract metric base (ndcg, recall, mrr) and type (@5, @10)
    df_ls_long["metric_base"] = df_ls_long["metric"].str.replace(r'_5|_10', '', regex=True)
    df_ls_long["metric_type"] = df_ls_long["metric"].str.extract(r'(5|10)').astype(int)

    # Define colors for each metric
    metric_colors = {'ndcg': '#fc6a03', 'recall': '#55bebd', 'mrr': '#e377c2'}

    # Define unique datasets and their subplot positions
    datasets = df_results_ls['dataset'].unique()
    num_datasets = len(datasets)

    # Create subplots: 1 row, 4 columns (one per dataset)
    fig = sp.make_subplots(
        rows=1, cols=num_datasets,
        subplot_titles=[f"{dataset}" for dataset in datasets],
        shared_yaxes=True  # Ensures consistent Y-axis scale
    )

    # Mapping dataset positions to subplot columns
    dataset_positions = {dataset: i+1 for i, dataset in enumerate(datasets)}

    # Plot data with overlapping bars for @5 and @10
    for _, row in df_ls_long.iterrows():
        dataset = row['dataset']
        metric = row['metric_base']
        metric_type = row['metric_type']
        value = row['value']
        
        col_idx = dataset_positions[dataset]  # Assign dataset to correct column
        color = metric_colors[metric]

        # Opacity for overlapping effect
        opacity = 0.6 if metric_type == 5 else 0.4  

        # Show text only for @10
        text_value = [value] if metric_type == 10 else None
        text_position = 'outside' if metric_type == 10 else None

        fig.add_trace(
            go.Bar(
                x=[metric],           
                y=[value],            
                marker=dict(color=color),
                opacity=opacity,
                text=text_value,        # 📌 Show values only for @10
                textposition=text_position
            ),
            row=1, col=col_idx
        )

    # Update layout
    fig.update_layout(
        title=f"Metrics across Datasets for the Best Approach: {approach_name}",
        height=400, width=1300,
        showlegend=False,  # No legend
        barmode="overlay"
    )

    # 📊 Ensure Y-axis ranges from 0 to 1 and show ticks in all subplots
    for col in range(1, num_datasets + 1):
        fig.update_yaxes(range=[0, 1], showticklabels=True, row=1, col=col)
        fig.update_xaxes(title_text="Metrics", row=1, col=col)

    fig.show()


def plot_table_results_all(df_results_ns, df_results_ss, df_results_ls, df_results_qr, df_results_q2d):
    df_ns_long = pd.melt(df_results_ns, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ss_long = pd.melt(df_results_ss, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ls_long = pd.melt(df_results_ls, id_vars=['dataset'], var_name='metric', value_name='value')
    df_qr_long = pd.melt(df_results_qr, id_vars=['dataset'], var_name='metric', value_name='value')
    df_q2d_long = pd.melt(df_results_q2d, id_vars=['dataset'], var_name='metric', value_name='value')
    df_ns_long['approach'] = 'NS'
    df_ss_long['approach'] = 'SS'
    df_ls_long['approach'] = 'LS'
    df_qr_long['approach'] = 'QR'
    df_q2d_long['approach'] = 'Q2D'
    df_long = pd.concat([df_ns_long, df_ss_long, df_ls_long, df_qr_long, df_q2d_long])

    # Extract metric base (ndcg, recall, mrr) and type (@5, @10)
    df_long["metric_base"] = df_long["metric"].str.replace(r'_5|_10', '', regex=True)
    df_long["metric_type"] = df_long["metric"].str.extract(r'(5|10)').astype(int)

    approach_colors = {
        'NS': '#fc6a03',  # Red for NS
        'SS': '#55bebd ',  # Green for SS
        'LS': '#e377c2',   # Blue for LS, 
        'QR': '#2ca02c',  # Green for QR
        'Q2D': '#9467bd'  # Purple for Q2D
    }

    # Define datasets and metrics
    datasets = df_results_ls['dataset'].unique()
    metrics = ['ndcg', 'recall', 'mrr']

    # Create subplots: 3 rows (metrics) × 4 columns (datasets)
    fig = sp.make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{dataset}" for metric in metrics for dataset in datasets],
        shared_yaxes=False
    )
    metric_positions = {'ndcg': 1, 'recall': 2, 'mrr': 3}
    dataset_positions = {dataset: i+1 for i, dataset in enumerate(datasets)}

    for _, row in df_long.iterrows():
        dataset = row['dataset']
        metric = row['metric_base']
        metric_type = row['metric_type']
        approach = row['approach']
        value = row['value']
        
        row_idx = metric_positions[metric]  # Metrics define the row
        col_idx = dataset_positions[dataset]  # Datasets define the column

        if metric_type == 10:
            fig.add_trace(
                go.Bar(
                x=[approach],  
                y=[value], 
                name=f"{metric.upper()} @{metric_type}",
                marker=dict(color=approach_colors[approach]),  # Color by approach
                opacity=0.6,
                textposition='outside',
                text=[value]  # Add the value as text on the bar
                ),
                row=row_idx, col=col_idx
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=[approach],  
                    y=[value], 
                    name=f"{metric.upper()} @{metric_type}",
                    marker=dict(color=approach_colors[approach]),  # Color by approach
                    opacity=0.6,
                    textposition='outside'  # Position text outside the bar
                ),
                row=row_idx, col=col_idx
            )

    # Update layout
    fig.update_layout(
        title="Comparison of Approaches per Dataset and Metric",
        height=1400, width=1400,
        bargap=0.1,  # Increase gap between bars to make them narrower
        barmode="overlay",
        xaxis_title="Approaches",
        yaxis_title="Metric Score",
        showlegend=False
    )
    # Update axis titles per row (corrected for metric names)
    for metric, row in metric_positions.items():
        fig.update_yaxes(title_text=f"{metric.upper()} Score", range=[0, 1], row=row, col=1)
        for col in range(1, len(datasets) + 1): 
            fig.update_xaxes(title_text="Approaches", row=row, col=col)

    for col in range(1, len(datasets) + 1): 
        fig.update_yaxes(range=[0, 0.9], row=1, col=col)
        fig.update_yaxes(range=[0, 1], row=2, col=col)
        fig.update_yaxes(range=[0, 0.8], row=3, col=col)

    fig.update_yaxes(range=[0, 0.6], row=1, col=3)
    fig.update_yaxes(range=[0, 0.6], row=2, col=3)
    fig.update_yaxes(range=[0, 0.6], row=3, col=3)
 
    fig.show()