# Nutriperso2018
This github is a repository for the code used in the Kantar 2014 data study for the INRA. This work was done under the supervision of profs. M. Sebag and P. Caillou.

The code is organized into 4 folders corresponding roughly to the introduction (preprocessing) + 3 parts of the report file. Although the 3 parts are somewhat independent from one another, it is necessary to run the preprocessing first, in order to put the data into an amenable format for any other part. Here, we will describe point by point each of the 4 code folders.

Before starting any analysis, note that the data folder is empty. Please put in the data folder the 2 following files: <br />
**1- achat_2014.csv (purchase data)**<br />
**2- menages_2014.csv (data about households)**

## I. Code Preprocessing

There are 5 files of interest, 4 that should be run in the following order + one of different tools and functions. <br />

### products_creation_table.py

* Inputs : achat_2014.csv
* Outputs : achat_2014cleaned.csv, produits_achats.csv<br />

The data does not include as such any information about products. This  information if hidden in the purchase data(achat_2014.csv). So each time a product appears in a purchase, it is followed by its full description.<br />

This fact is suboptimal and wastes a lot of memory and computations. Our first task will be to extract from the purchase table a table describing products, and also to output a pruned version of the purchases, without all the redundant information about products (we will only keep the product code).

### foyers preprocessing.py

Inputs : menages_2014.csv__<br />
Outputs : foyers_traites.csv, foyers_dedupliques.csv

In the first file, each line represents on responsible member of the household (father, mother). Each household gets thus represented as at most 2 data points. 2 members of the same household will share some features (revenue, geography, etc.) but not others (gender, age, education, etc.)<br />

In the Second one, it is the same data but same where households have not been doubled. So there will be for each household entries such as age_mother, age_father, etc.). For single families, such entries will be reported as NA. <br />

### products_clustering_table.py
Inputs : produits_achats.csv<br />
Outputs : Clutering mano.csv, Clustering_auto.csv

Clutering mano : clustering of products using manually selected relevant features for each category. The clustering yields circa 4,000 product-clusters. It outputs the file cluster_products_mano.csv<br />

Clustering_auto.csv : clustering of features using an automatic selection of features, here features that are present for more than 50% of the products, and where quantitative features are put in quartiles.<br />

Note, a framework for refining the manual clustering is available in the file YYY, and the csv ZZZ contains the kept variables for each subgroup. If you want to modify which variables are kept in a subgroup, please see the tool_clustering.py file, and modifiy accordingly the dictionary.

### create_table_purchases.py

Inputs : achat_2014cleaned.csv<br />
Outputs : purchase_table_full, purchase_table_full_weekly, purchase_table_full_monthly, purchase_table_codepanier

This code serves to create from the initial purchase data (where each line is a purchase) the full purchase matrix. This comes in the form of a sparse matrix – which is in our opinion the best way to store the data. Thus, each time 3 files will be created : the purchase matrix itself, the names of the rows, and the names of the columns. Although according to the granularity of the data, the name of the rows will change, the name of the columns (products) should stay the same, i.e. be the 170,000 product codes. Here are the different options for the granularity:<br />

- Yearly aggregation: each row is a household, and the row-vectors represent the yearly purchases of the household (purchase_table_full)<br />

-Monthly / weekly aggregation : each row is a household and a month, and the row-vectors represent the monthly / weekly purchases of the household (purchase_table_full_weekly, purchase_table_full_monthly)
<br />
- Basket aggregation: this is the finest level of granularity we have achieved. Each row is a basket, combining all the products one houshehold has purchased in one time (purchase_table_codepanier).<br />

The data is saved in the npz format, a format especially suitable for sparse matrices.

