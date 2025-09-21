from django.shortcuts import render
from .ml_model import ModelManager

# Create your views here.

def landing(request):
    """Landing page with CTA to go to the prediction page."""
    return render(request, 'landing.html')


def predict_price(request):
    prediction = None           # predicted price per unit area (from model)
    total_price = None          # predicted total house price
    error = None

    if request.method == 'POST':
        try:
            size_sqft = float(request.POST.get('size'))
            bedrooms = int(request.POST.get('bedrooms'))
            age = float(request.POST.get('age'))

            # Basic validation (keep same ranges)
            if not (1000 <= size_sqft <= 5000):
                error = "House size must be between 1000 and 5000 sq ft."
            elif not (1 <= bedrooms <= 5):
                error = "Number of bedrooms must be between 1 and 5."
            elif not (0 <= age <= 50):
                error = "House age must be between 0 and 50 years."
            else:
                # Convert square feet to square meters
                size_sqm = size_sqft * 0.092903

                # Use ModelManager to predict price per unit area (expects size in sqm)
                predictor = ModelManager()
                prediction = predictor.predict(size_sqm, bedrooms, age)

                # Compute total house price = predicted price per unit area * area (sqm)
                total_price = round(prediction * size_sqm, 2)
        except (ValueError, TypeError):
            error = "Invalid input. Please enter valid numbers."

    return render(request, 'predict.html', {
        'prediction': prediction,
        'total_price': total_price,
        'error': error,
    })
