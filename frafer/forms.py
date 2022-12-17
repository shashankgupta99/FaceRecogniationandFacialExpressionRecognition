from django.forms import Form, CharField

class UserForm(Form):
    username=CharField(max_length=50)
    password=CharField(max_length=50)
    name=CharField(max_length=50)
    email=CharField(max_length=50)
    mobile=CharField(max_length=50)