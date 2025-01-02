
from django.contrib import admin
from django.urls import path
from DisasterTweet.views import tweet_classification  
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', tweet_classification, name='home'),  
    path('tweet-classification/', tweet_classification, name='tweet_classification'),  
    path('classify_tweet/', views.classify_tweet, name='classify_tweet'),
]
