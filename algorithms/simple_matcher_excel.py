import pandas as pd

def simple_column_matcher(df1, df2, key_column, excel_path=None):
    """Row-by-row comparison with Excel export capability"""
    
    # Ensure perfect alignment using key column
    aligned = pd.merge(df1, df2, on=key_column, suffixes=('_1', '_2'))
    total_rows = len(aligned)
    
    matches = {}
    
    # Compare columns pairwise
    for col1 in df1.columns:
        if col1 == key_column:
            continue
            
        best_count = 0
        best_match = None
        
        for col2 in df2.columns:
            if col2 == key_column:
                continue
                
            # Row-by-row comparison
            match_count = 0
            for i in range(len(aligned)):
                if aligned[col1].iloc[i] == aligned[col2].iloc[i]:
                    match_count += 1
                    
            if match_count > best_count:
                best_count = match_count
                best_match = col2
                
        # Add percentage calculation
        matches[col1] = {
            'best_match': best_match,
            'match_count': best_count,
            'match_percentage': f"{(best_count/total_rows)*100:.2f}%"
        }
    
    # Export to Excel if path is provided
    if excel_path:
        # Create DataFrame from results
        results_df = pd.DataFrame.from_dict(matches, orient='index')
        results_df.reset_index(inplace=True)
        results_df.columns = ['DF1 Column', 'Best DF2 Match', 'Match Count', 'Match Percentage']
        
        # Write to Excel with formatting
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Format headers
            for col_num, value in enumerate(results_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 20)
            worksheet.set_column('B:B', 20)
            worksheet.set_column('C:C', 15)
            worksheet.set_column('D:D', 15)
    
    return matches

# Usage example:
# result = simple_column_matcher(df1, df2, 'ID', excel_path='matches.xlsx')