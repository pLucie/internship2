#this function will load all the required packages for the project.

#This function checks if the packages are already installed and if not will install and load them, otherwise only load them.
package_check <- function(pkg){
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

#loads the devtools package to download forestforesight package.
package_check("devtools")

#install ForestForesight Package from github
devtools::install_github("ForestForesight/ForestForesight")

#load all the packages
package_check("ForestForesight")
package_check("terra")
package_check("sf")
package_check("reticulate")
