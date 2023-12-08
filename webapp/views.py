from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

# Add the necessary imports for recommendation functionality
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the styles.csv and images.csv files
file_path_csv = 'images.csv'
file_path_excel = 'styles.csv'  # Assuming it's an Excel file, change the extension accordingly
dfi = pd.read_csv(file_path_csv,nrows = 10000)
dfs = pd.read_csv('styles.csv', error_bad_lines=False, nrows = 10000)

# Preprocess the text data for recommendation
dfs['combined_text'] = dfs[['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName']].astype(str).agg(' '.join, axis=1)
dfs['combined_text'] = dfs['combined_text'].apply(lambda text: re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()))

# Vectorize the text data for recommendation
vectorizer = TfidfVectorizer(stop_words='english')
text_matrix = vectorizer.fit_transform(dfs['combined_text'])
cosine_sim = cosine_similarity(text_matrix, text_matrix)

# Recommendation function
def get_recommendations(custom_text):
    processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', custom_text.lower())
    custom_text_vector = vectorizer.transform([processed_text])
    sim_scores = cosine_similarity(custom_text_vector, text_matrix).flatten()
    top_indices = sim_scores.argsort()[:-6:-1]
    recommended_products = dfs.loc[top_indices, ['id', 'productDisplayName']]
    return recommended_products

# Views
def index(request):
    template = loader.get_template('webapp/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

@csrf_exempt
def recommend(request):
    if request.method == 'POST':
        custom_text = request.POST.get('custom_text', '')
        recommended_products = get_recommendations(custom_text)

        recommended_ids = recommended_products['id'].tolist()
        recommended_filenames = [str(id_) + '.jpg' for id_ in recommended_ids]
        recommended_images = pd.merge(dfi, pd.DataFrame(recommended_filenames, columns=['filename']), how='inner', on='filename')['link'].tolist()

        recommendations = []
        for i, (image_url, product_description) in enumerate(zip(recommended_images, recommended_products['productDisplayName'])):
            recommendation = {
                'id': i ,
                'product_description': product_description,
                'image_url': image_url
            }
            recommendations.append(recommendation)

        return JsonResponse({'recommended_products': recommendations})
    else:
        return HttpResponse(status=400)

