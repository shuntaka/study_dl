# set up for colab
'''
!curl https: // course.fast.ai/setup/colab | bash
'''
cut = train_df['Date'][(
    train_df['Date'] == train_df['Date'][len(test_df)])].index.max()
