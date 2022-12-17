from django.shortcuts import render

from frafer.facialexpression import predictexpression
from frafer.forms import UserForm
from frafer.facedetection import verify_user, capture_img, trainimg
from frafer.models import RegistrationModel


def registration(request):

    # Get the posted form
    registrationForm = UserForm(request.POST)

    print("in function")
    if registrationForm.is_valid():

        regModel = RegistrationModel()

        regModel.name = registrationForm.cleaned_data["name"]
        regModel.email = registrationForm.cleaned_data["email"]
        regModel.mobile = registrationForm.cleaned_data["mobile"]
        regModel.username = registrationForm.cleaned_data["username"]
        regModel.password = registrationForm.cleaned_data["password"]

        user = RegistrationModel.objects.filter(username=regModel.username).first()

        if user is not None:
            return render(request, 'registration.html', {"message": user.username})
        else:
            regModel.save()
            capture_img(registrationForm.cleaned_data["username"])
            trainimg()
            return render(request, 'registration.html', {"message": "Registred Sucessfully"})
    else:
        print("in else")
        return render(request, 'registration.html', {"message": "Invalid Form"})

def login(request):

    uname = request.GET["username"]
    upass = request.GET["password"]

    user = RegistrationModel.objects.filter(username=uname, password=upass).first()

    if user is not None:
        id = verify_user()
        print(id,uname)
        if str(id)==uname:
            request.session['username'] = uname
            return render(request, 'home.html')
        else:
            return render(request, 'index.html', {"message": "Face Does not Match"})
    else:
        return render(request, 'index.html', {"message": "Invalid username or Password"})

def findfacialexpression(request):
    if request.method == "GET":
        predictexpression(request.session['username'])
    return render(request, 'index.html', {"message": "Invalid Request"})

def logout(request):
    try:
        del request.session['username']
    except:
        pass
    return render(request, 'index.html', {})
