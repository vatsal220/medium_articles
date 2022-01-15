import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# constants
# source = https://docs.google.com/spreadsheets/d/19h0WT9Xjf9ncNmJDWpd8fzz7YnB37nemDnxKSO0EE8o/edit#gid=1664289965
file_path = './data/anon_salary.csv'

def clean_currency(cur):
    '''
    Will clean the currency column 
    '''
    curr_dct = {
        # I'm aware that dollars and $ can be across many different currencies, I'm making a generalized assumption
        'usd' : ['us dollar', 'us dollars', 'usda', 'usdp', 'dollars', '$', 'usa', 'uds', 'ysd', 'uss'],  
        'cad' : ['can', 'canadian', 'caf', 'canada'],
        'eur' : ['euro', 'euros', 'eu',],
        'gbp' : ['pound', 'pound sterling', 'pounds', 'british pounds', 'british pound', 'gbp £']
    }
    
    cur = str(cur).lstrip().rstrip().strip().lower()
    for k,v in curr_dct.items():
        if cur in v:
            return k

    if cur == 'nan' or cur == 'n/a' or cur == '-':
        return np.nan
    return cur

def clean_wage(wage):
    '''
    Will clean the user input wage to be an integer
    '''
    wage=str(wage).lower()
    wage = wage.replace('$','').replace(' ', '').replace(',', '').replace('cad', '').replace('usd','').replace('()','').replace('k', '000')
    try:
        return int(wage)
    except:
        return np.nan
    
def clean_org(org):
    '''
    This will clean the organization section. It will return np.nan if the user has not provided
    their organization, otherwise it will return the lowered organization passed
    '''
    org_dct = {
        'self employed' : ['self-employed', 'self'],
        'meta' : ['meta', 'facebook', 'face book', 'fb'],
        np.nan : ['na', 'n/a', 'none', 'prefer not to say', 'N/a', '-', 'nan', 'anon', 'anonymous', 'Prefer not to say']
    }
    
    org = str(org).lower().lstrip().rstrip()
    for k,v in org_dct.items():
        if org in v:
            return k
    return org

def clean_loc_city(city):
    city_dct = {
        'nyc' : ['new york, ny', 'nyc', 'new york', 'new york city, ny', 'new york city', 'new york, new york', 'ny', 'ny ny', 'new york new york', 'new york ny'],
        'la' : ['los angeles, ca', 'los angeles'],
        np.nan : ['nan', 'none', '-', 'n/a'],
        'toronto' : ['toronto', 'toronto, ontario', 'toronto, canada', 'toronto, on', 'toronto, ca'],
        'sf' : ['san francisco ca', 'sf', 'sf ca', 'sf, ca', 'sf, california', 'san francisco, ca', 'sf', 'san fransisco', 'san francisco']
    }
    city = str(city).lower().lstrip().rstrip()
    for k,v in city_dct.items():
        if city in v:
            return k
    cleaned_city = str(city).split(',')[0]
    return cleaned_city.lower().lstrip().rstrip()

def clean_country(country):
    country_dct = {
        'usa' : [
            'united stayes', 'united statea', 'united stares', 'united states', 'unites states', 'united state', 
            'california', 'new york', 'us', 'ny', 'united statws',  'usa', 'united states', 'united states of america',
            'america', 'u.s.a.', 'u.s.a', 'u.s', 'u.s.', 'ny', 'united stated', 'los angeles', 'untied states',
            'united statss', 'united sates', 'la', 'sacramento', 'ysa'
        ],
        'canada' : ['canada', 'ca', 'can'],
        'united kingdom' : ['united kingdom', 'uk', ''],
        np.nan : ['nan', 'n/a', 'none', '-']
    }
    country = str(country).lower().lstrip().rstrip()
    for k,v in country_dct.items():
        if country in v:
            return k
    return country

def clean_job(job):
    job_dct = {
        'swe' : ['swe', 'software engineer', ''],
        'nurse' : ['rn', 'registered nurse', 'nurse'],
        'data scientist' : ['data science', 'data scientist'],
        np.nan : ['nan', '-', 'none']
    }
    job = str(job).lstrip().rstrip().lower()
    for k,v in job_dct.items():
        if job in v:
            return k
    return job

def clean_gender(gender):
    gender_dct = {
        'female' : ['female'],
        'male' : ['male'],
        np.nan : ['prefer not to say', 'n/a', '-', "you shouldn't have included this free text field eh?"],
        'non-binary' : ['non-binary', 'non binary', 'nb', 'nonbinary'],
        'trans' : ['trans', 'transgender fem', 'genderqueer, transmasculine', 'trans female', 'trans man', 'trans male']
    }
    gender = str(gender).lower().lstrip().rstrip()
    for k,v in gender_dct.items():
        if gender in v:
            return k
        if 'binary' in gender:
            return 'non-binary'
    return 'other'

def clean_yoe(yoe):
    '''
    This function will convert the yoe to their associated integer or leave it as
    `20+` or return np.nan
    '''
    yoe = str(yoe).lstrip().rstrip()
    if yoe == 'nan':
        return np.nan
    elif yoe == '20+':
        return '20+'
    else:
        try:
            return int(yoe)
        except:
            return yoe
        
def clean_sick_days(days):
    sick_dct = {
        'unlim' : ['as many as i want', 'as many as i need', 'infinite', 'unlimited', '🚩unlimited', 'no limit', 'pto', 'unlimited (kind of)', 'unlimited pto', '"unlimited"'],
        np.nan : ['none', 'nan', 'n/a', '-', '?', 'na', 'idk', 'not sure', 'unsure'],
        10 : ['2 weeks', '10 days', '14 days'],
        5 : ['1 week', '5 days', '7 days', '40 hours'],
        15 : ['3 weeks', '15 days', '21 days'],
        20 : ['4 weeks', '20 days'],
        25 : ['5 weeks', '25 days'],
        30 : ['30 days', '6 weeks']
    }
    days = str(days).lower().lstrip().rstrip().replace('"', '').replace('"', '')
    for k,v in sick_dct.items():
        if days in v:
            return k
    try:
        return int(days)
    except:
        return days

def clean_open_discussions(disc):
    disc_dct = {
        'yes' : ['yes', 'ya', 'yeah'],
        'no'  : ['no', 'nope', 'never']
    }
    
    disc = str(disc).lower().lstrip().rstrip()
    for k,v in disc_dct.items():
        for val in v:
            if val in disc:
                return k
    return 'other'

def clean_ethnicity(eth):
    eth_dct = {
        'caucasian' : ['white', 'caucasian', 'white female', 'white male'],
        'african american' : ['black', 'african', 'black american', 'black/african american', 'african american', 'black woman', 'african-american', 'black male', 'black man', 'black female'],
        'hispanic' : ['hispanic', 'latina', 'latino', 'latinx', 'mexican-american', 'latino/hispanic',  'mexican', 'mexico', 'mexican american', 'latin'],
        'asian' : ['asian', 'asian american', 'south asian', 'asian-american', 'korean', 'indian american', 'vietnamese', 'east asian', 'asian indian', 'asian woman', 'filipino', 'filipina', 'indian', 'chinese'],
        'middle eastern' : ['middle eastern', 'arab', 'pakistani'],
        'native' : ['native american', 'native', 'indigenous']
    }
    eth = str(eth).lower().lstrip().rstrip()
    
    for k,v in eth_dct.items():
        if eth in v:
            return k
    return 'unknown'

def clean_mat_leave(mat):
    mat_dct = {
        np.nan : ['none', 'not sure', 'unsure','no', 'no idea', '?', 'idk', 'unknown', 'n/a', '-', "don't know", '0'],
        0.5 : ['2 weeks'],
        1 : ['1', '1 month'],
        1.5 : ['6 weeks', '1.5'],
        2 : ['2', '2 months', '8 weeks'],
        3 : ['3', '3 months', '12 weeks'],
        4 : ['4', '4 months', '16 weeks'],
        5 : ['5', '5 months', '20 weeks'],
        6 : ['6', '6 months', '24 weeks'],
        float('inf') : ['unlimited']
    }
    mat = str(mat).lower().lstrip().rstrip()
    for k,v in mat_dct.items():
        if mat in v:
            return k
    return 'unknown'
    
def clean_industry(ind):
    ind_dct = {
        'tech' : ['tech'],
        'finance' : ['fin tech', 'finance', 'fintech', 'financial services'],
        'heath care' : ['healthcare', 'health care', 'HospTechalTechy', 'healthcare tech'],
        'non profit' : ['nonproftech', 'non proftech', 'non profit']
    }
    ind = str(ind).lower().lstrip().rstrip()
    for k,v in ind_dct.items():
        if ind in v:
            return k
    return ind

def visualize_group(d, remove_vals, group_col, agg_col, title, th, n):
    '''
    Given a dataframe d, this function will filter the values and visualize the results
    based on the group and aggregate columns.
    
    params:
        d (DataFrame) : The dataframe with the data you're visualizing
        remove_vals (List) : The list of values you want to filter out of the input df
        group_col (String) : The column you want to group
        agg_col (String) : The column you want to aggregate the result of
        title (String) : The title of the plot
        th (Integer) : The threshold of values each group must have
        n (Integer) : The number of results you want to filter
        
    example:
        visualize_group(
            d = us_df,
            remove_vals = None, 
            group_col = 'job_title',
            agg_col = 'annual_wage',
            title = 'Top 15 Professions by Median Annual Wage',
            th = 3,
            n = 15
        )
    '''
    if remove_vals:
        d = d[~d[agg_col].isin(remove_vals)]
    
    df = d.groupby([group_col])[agg_col].agg(['count', 'mean', 'median']).reset_index().rename(columns = {
        'count' : group_col + '_count', 'mean' : 'avg_' + agg_col, 'median' : 'median_' + agg_col
    }).copy()
    
    df = df[df[group_col + '_count'] > th].sort_values(by = 'median_' + agg_col, ascending = False).copy()
    if n:
        df = df.head(n)
        
    plt.clf()
    plt.barh(y = df[group_col].values, width = df['median_' + agg_col])
    plt.ylabel(group_col)
    plt.xlabel("median_" + agg_col)
    plt.title(title)
    plt.show()
    
def main():
    # import data
    d = pd.read_csv(file_path)
    
    # clean data
    rename_cols = {
        'Timestamp' : 'timestamp',
        'Country' : 'country',
        'Age Range' : 'age_range',
        'Highest Level of Education Received' : 'max_edu',
        'Company Name' : 'org',
        'Years of Experience' : 'yoe',
        'Closest Major City and State (e.g. Santa Clara, CA)' : 'loc_city',
        'Annual Base Salary (if hourly, please convert to annual)' : 'annual_wage',
        'How many vacation days are you given per year?' : 'vacation_days_yearly',
        'How many sick days are you given per year?' : 'sick_days_yearly',
        'Do you openly discuss salary with your colleagues?' : 'open_wage_discussions',
        'How many months Maternity or Paternity does your company offer?' : 'maternity_leave_months',
        'Diverse Identity (Optional)' : 'ethnicity',
        'Currency (USD, CAD, etc)' : 'currency',
        'Gender (optional)' : 'gender',
        'Annual Bonus' : 'bonus',
        'Annual Average of RSUs' : 'avg_rsu',
        'Signing Bonus (if none, leave blank)' : 'signing_bonus',
        'Job Title' : 'job_title',
        'How many days per week are you required to work onsite/in the office?' : 'days_per_week_in_office'
    }

    d.rename(columns = rename_cols, inplace = True)
    d.drop(columns = ['Unnamed: 20'], inplace = True)
    
    d['currency'] = d['currency'].apply(lambda x : clean_currency(x))
    d['annual_wage'] = d['annual_wage'].apply(clean_wage).astype(float)
    d['signing_bonus'] = d['signing_bonus'].apply(clean_wage).astype(float)
    d['bonus'] = d['bonus'].apply(clean_wage).astype(float)
    d['avg_rsu'] = d['avg_rsu'].apply(clean_wage).astype(float)
    d['org'] = d['org'].apply(clean_org)
    d['loc_city'] = d['loc_city'].apply(clean_loc_city)
    d['country'] = d['country'].apply(clean_country)
    d['job_title'] = d['job_title'].apply(clean_job)
    d['gender'] = d['gender'].apply(clean_gender)
    d['yoe'] = d['yoe'].apply(clean_yoe)
    d['sick_days_yearly'] = d['sick_days_yearly'].apply(clean_sick_days)
    d['vacation_days_yearly'] = d['vacation_days_yearly'].apply(clean_sick_days)
    d['open_wage_discussions'] = d['open_wage_discussions'].apply(clean_open_discussions)
    d['ethnicity'] = d['ethnicity'].apply(clean_ethnicity)
    d['maternity_leave_months'] = d['maternity_leave_months'].apply(clean_mat_leave)
    d['Industry'] = d['Industry'].apply(clean_industry)
    
    # analysis & visualizations
    us_df = d[d['country'] == 'usa'].copy()
    ca_df = d[d['country'] == 'canada'].copy()
    print(us_df.shape, ca_df.shape)
    
    cad_usd_conversion_rate = 0.8
    ca_df['annual_wage'] = ca_df['annual_wage'] * cad_usd_conversion_rate
    
    # Jobs with the Highest Median Salary
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'job_title',
        agg_col = 'annual_wage',
        title = 'Top 15 Professions by Median Annual Wage - America',
        th = 5,
        n = 15
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'job_title',
        agg_col = 'annual_wage',
        title = 'Top 15 Professions by Median Annual Wage - Canada',
        th = 5,
        n = 15
    )
    
    # Industries with the Highest Median Salary
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'Industry',
        agg_col = 'annual_wage',
        title = 'Top 15 Industries by Median Annual Wage - America',
        th = 5,
        n = 15
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'Industry',
        agg_col = 'annual_wage',
        title = 'Top 15 Industries by Median Annual Wage - Canada',
        th = 5,
        n = 15
    )
    
    # Average Income per YoE
    ca_df['yoe'] = ca_df['yoe'].apply(lambda x : [20 if x == '20+' else x][0]).astype(float)
    us_df['yoe'] = us_df['yoe'].apply(lambda x : [20 if x == '20+' else x][0]).astype(float)

    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'yoe',
        agg_col = 'annual_wage',
        title = 'Median YoE per Year of Experience - America',
        th = 5,
        n = 20
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'yoe',
        agg_col = 'annual_wage',
        title = 'Median YoE per Year of Experience - Canada',
        th = 5,
        n = 20
    )
    
    # Median Wages per Gender
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'gender',
        agg_col = 'annual_wage',
        title = 'Median Wage per Gender - America',
        th = 5,
        n = 20
    )
    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'gender',
        agg_col = 'annual_wage',
        title = 'Median Wage per Gender - Canada',
        th = 5,
        n = 20
    )
    
    # Median Wages per Ethnic Group  
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'ethnicity',
        agg_col = 'annual_wage',
        title = 'Median Wage per Ethnicity - America',
        th = 5,
        n = 20
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'ethnicity',
        agg_col = 'annual_wage',
        title = 'Median Wage per Ethnicity - Canada',
        th = 5,
        n = 20
    )
    
    # Median Wages per Age Group
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'age_range',
        agg_col = 'annual_wage',
        title = 'Median Income per Age Group - America',
        th = 5,
        n = None
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'age_range',
        agg_col = 'annual_wage',
        title = 'Median Income per Age Group - Canada',
        th = 5,
        n = None
    )
    
    # Highest Paying Organizations
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'org',
        agg_col = 'annual_wage',
        title = 'Highest Paying Organizations - America',
        th = 5,
        n = 20
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'org',
        agg_col = 'annual_wage',
        title = 'Highest Paying Organizations - Canada',
        th = 5,
        n = 20
    )
    
    # Median Wages per Maximum Education Level
    visualize_group(
        d = us_df,
        remove_vals = None, 
        group_col = 'max_edu',
        agg_col = 'annual_wage',
        title = 'Median Wages per Maximum Education Level - America',
        th = 5,
        n = None
    )

    visualize_group(
        d = ca_df,
        remove_vals = None, 
        group_col = 'max_edu',
        agg_col = 'annual_wage',
        title = 'Median Wages per Maximum Education Level - Canada',
        th = 5,
        n = None
    )
    
if __name__ == '__main__':
    main()