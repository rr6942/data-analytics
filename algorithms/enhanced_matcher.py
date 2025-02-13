import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def enhanced_column_matcher(df1, df2, key_column, excel_path=None, fuzzy_threshold=80):
    """Enhanced column matcher with fuzzy matching, data type checks, and statistical similarity."""
    
    # Handle missing values
    df1 = df1.fillna('MISSING')
    df2 = df2.fillna('MISSING')
    
    # Ensure perfect alignment using key column
    aligned = pd.merge(df1, df2, on=key_column, suffixes=('_1', '_2'))
    total_rows = len(aligned)
   
    matches = {}
    
    # Compare columns pairwise
    for col1 in df1.columns:
        if col1 == key_column:
            continue
            
        best_match = None
        best_score = 0
        
        for col2 in df2.columns:
            if col2 == key_column:
                continue
                
            # Fuzzy match column names
            fuzzy_score = fuzz.ratio(col1.lower(), col2.lower())
            if fuzzy_score < fuzzy_threshold:
                continue
                
            # Check data types
            if not compare_data_types(df1, df2, col1, col2):
                continue
                
            # Calculate cosine similarity
            similarity = calculate_cosine_similarity(aligned[col1], aligned[col2])
            
            if similarity > best_score:
                best_score = similarity
                best_match = col2
                
        # Add percentage calculation
        matches[col1] = {
            'best_match': best_match,
            'similarity_score': f"{best_score:.2f}",
            'match_percentage': f"{(best_score * 100):.2f}%"
        }
    
    # Export to Excel if path is provided
    if excel_path:
        results_df = pd.DataFrame.from_dict(matches, orient='index')
        results_df.reset_index(inplace=True)
        results_df.columns = ['DF1 Column', 'Best DF2 Match', 'Similarity Score', 'Match Percentage']
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            for col_num, value in enumerate(results_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 20)
            worksheet.set_column('C:C', 15)
            worksheet.set_column('D:D', 15)
    
    return matches

# Helper functions
def compare_data_types(df1, df2, col1, col2):
    return df1[col1].dtype == df2[col2].dtype

def calculate_cosine_similarity(col1, col2):
    vectorizer = CountVectorizer().fit_transform(col1.astype(str) + " " + col2.astype(str))
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

# Usage example:
# result = enhanced_column_matcher(df1, df2, 'ID', excel_path='enhanced_matches.xlsx')