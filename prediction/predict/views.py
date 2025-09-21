from django.shortcuts import render
from .ml_model import ModelManager

# Create your views here.

def predict_price(request):
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            size = float(request.POST.get('size'))
            bedrooms = int(request.POST.get('bedrooms'))
            age = float(request.POST.get('age'))
            if not (1000 <= size <= 5000):
                error = "House size must be between 1000 and 5000 sq ft."
            elif not (1 <= bedrooms <= 5):
                error = "Number of bedrooms must be between 1 and 5."
            elif not (0 <= age <= 50):
                error = "House age must be between 0 and 50 years."
            else:
                predictor = ModelManager()
                prediction = predictor.predict(size, bedrooms, age)
        except (ValueError, TypeError):
            error = "Invalid input. Please enter valid numbers."
    return render(request, 'predict.html', {'prediction': prediction, 'error': error})
