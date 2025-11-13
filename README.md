**News Article Scraping and Summarization**

This repository provides tools to scrape articles from Malaysian news sites (Borneo Post, Malay Mail, The Star), summarize them using transformer models, and serve results through a Streamlit app.
**Features**
- Scrape articles from multiple news websites using Selenium and Scrapy.
- Summarize articles using T5 transformer models.
- Store scraped and summarized data as CSV.
- Upload results to AWS S3.

Interactive Streamlit UI for scraping and summarizing tasks.

**Setup**
Install dependencies:
Run pip install -r requirements.txt in your repo folder.

**AWS Credentials:**
Update your AWS access and secret keys in model-building scripts before uploading to S3.

**Usage**
- Run streamlit run streamlit_app.py to start the UI.
- Use the sidebar to select between Scrape Articles and Summarize Articles.
- Choose a website to scrape, then run the summarization once scraping is done.
- Summarized articles will be saved and uploaded to S3.

**File Structure**
streamlit_app.py: Main GUI application for user interaction.​

Borneo.py, Scrapy2.py, Star.py: Scraping scripts for respective news sites.​

Borneo_ModelBuilding.py, ModelBuilding2.py, Star_ModelBuilding.py: Summarization scripts for each site.​



