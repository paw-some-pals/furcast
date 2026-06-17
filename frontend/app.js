let currentAnimal = null;

const options = {
    dog: {
        breed: [
            'Afghan Hound',
            'Akita',
            'Alaskan Klee Kai',
            'Alaskan Malamute',
            'American Bulldog',
            'American Eskimo Dog',
            'American Hairless Terrier',
            'American Leopard Hound',
            'American Staffordshire Terrier',
            'American Water Spaniel',
            'Anatolian Shepherd Dog',
            'Australian Cattle Dog',
            'Australian Shepherd',
            'Barbet',
            'Basenji',
            'Basset Hound',
            'Beagle',
            'Beauceron',
            'Belgian Malinois',
            'Bergamasco Sheepdog',
            'Berger Picard',
            'Bernese Mountain Dog',
            'Bichon Frise',
            'Bloodhound',
            'Boerboel',
            'Bolognese',
            'Border Collie',
            'Borzoi',
            'Boston Terrier',
            'Boxer',
            'Bracco Italiano',
            'Brittany',
            'Bull Terrier',
            'Bullmastiff',
            'Cane Corso',
            'Cardigan Welsh Corgi',
            'Cavalier King Charles Spaniel',
            'Chihuahua',
            'Chinese Shar-Pei',
            'Chinook',
            'Chow Chow',
            'Cocker Spaniel',
            'Coton de Tulear',
            'Dachshund',
            'Dalmatian',
            'Doberman Pinscher',
            'Field Spaniel',
            'French Bulldog',
            'German Longhaired Pointer',
            'Giant Schnauzer',
            'Golden Retriever',
            'Great Dane',
            'Great Pyrenees',
            'Greyhound',
            'Havanese',
            'Hovawart',
            'Irish Setter',
            'Irish Terrier',
            'Italian Greyhound',
            'Japanese Chin',
            'Jindo',
            'Keeshond',
            'Komondor',
            'Labrador Retriever',
            'Maltese',
            'Miniature Pinscher',
            'Mudi',
            'Newfoundland',
            'Norfolk Terrier',
            'Nova Scotia Duck Tolling Retriever',
            'Otterhound',
            'Papillon',
            'Pekingese',
            'Pembroke Welsh Corgi',
            'Plott Hound',
            'Pomeranian',
            'Poodle (Miniature)',
            'Pug',
            'Pumi',
            'Rhodesian Ridgeback',
            'Rottweiler',
            'Russian Toy',
            'Samoyed',
            'Schipperke',
            'Shetland Sheepdog',
            'Shiba Inu',
            'Shih Tzu',
            'Siberian Husky',
            'Smooth Fox Terrier',
            'Staffordshire Bull Terrier',
            'Tibetan Mastiff',
            'Treeing Walker Coonhound',
            'Vizsla',
            'West Highland White Terrier',
            'Whippet',
            'Xoloitzcuintli',
            'Yorkshire Terrier',
        ],
        colour: ['Multi/Unknown', 'Brindle', 'Black', 'Brown/Chocolate', 'Tan/Yellow/Red', 'Gray/Blue', 'White/Cream']
    },
    cat: {
        breed: [
            'Abyssinian',
            'Aegean',
            'American Bobtail',
            'American Longhair',
            'American Shorthair',
            'American Wirehair',
            'Aphrodite Giant',
            'Arabian Mau',
            'Asian',
            'Australian Mist',
            'Bambino',
            'Bengal Cats',
            'Birman',
            'Bombay',
            'Brazilian Shorthair',
            'British Longhair',
            'British Shorthair',
            'Burmese',
            'Burmilla',
            'California Spangled',
            'Chantilly-Tiffany',
            'Chausie',
            'Colorpoint Shorthair',
            'Cornish Rex',
            'Cyprus',
            'Devon Rex',
            'Donskoy',
            'European Shorthair',
            'Foldex',
            'German Rex',
            'Highlander',
            'Japanese Bobtail',
            'Javanese',
            'Khao Manee',
            'Kurilian Bobtail',
            'Lykoi',
            'Maine Coon',
            'Manx',
            'Mekong Bobtail',
            'Nebelung',
            'Oriental Bicolor',
            'Persian',
            'Peterbald',
            'Pixie-Bob',
            'Ragdoll Cats',
            'Russian Blue',
            'Savannah',
            'Scottish Fold',
            'Serengeti',
            'Siamese Cat',
            'Siberian',
            'Singapura',
            'Snowshoe',
            'Sokoke',
            'Somali',
            'Sphynx',
            'Tonkinese',
            'Toyger',
            'Turkish Angora',
            'Turkish Van',
            'Ukrainian Levkoy',
            'York Chocolate',
        ],
        colour: ['Multi/Unknown', 'Point/Lynx', 'Tabby/Tortie', 'Gray/Blue', 'Orange/Red', 'White', 'Black']
    }
};

function chooseAnimal(species) {
    currentAnimal = species;

    document.getElementById('animal_species').value = species;
    document.getElementById('form-emoji').textContent = species === 'cat' ? '🐱' : '🐶';
    document.getElementById('form-title').textContent = species === 'cat' ? 'Cat Details' : 'Dog Details';

    populateSelect('breed-select', options[species].breed);
    populateSelect('breed2-select', options[species].breed);
    populateSelect('colour-select', options[species].colour);

    document.getElementById('mixed-checkbox').checked = false;
    document.getElementById('breed2-section').classList.add('hidden');

    const sizeSection = document.getElementById('size-section');
    if (species === 'dog') {
        sizeSection.classList.remove('hidden');
    } else {
        sizeSection.classList.add('hidden');
        document.getElementById('size-select').value = '';
    }

    showPage('page-form');
}

document.getElementById('mixed-checkbox').addEventListener('change', function () {
    const breed2Section = document.getElementById('breed2-section');
    if (this.checked) {
        breed2Section.classList.remove('hidden');
    } else {
        breed2Section.classList.add('hidden');
        document.getElementById('breed2-select').value = '';
    }
});

function populateSelect(id, items) {
    const sel = document.getElementById(id);
    sel.innerHTML = '<option value="">-- Select --</option>';
    items.forEach(function (val) {
        const opt = document.createElement('option');
        opt.value = val;
        opt.textContent = val;
        sel.appendChild(opt);
    });
}

function showPage(id) {
    document.querySelectorAll('.page').forEach(function (p) {
        p.classList.add('hidden');
    });
    document.getElementById(id).classList.remove('hidden');
}

function goBack(pageId) {
    showPage(pageId);
}

document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    document.getElementById('error-msg').style.display = 'none';

    const form = e.target;

    const required = [
        form.city.value,
        form.age_intake.value,
        form.sex.value,
        form.spay_neuter.value,
        form.intake_month.value,
        form.intake_day.value,
        form.intake_year.value,
        form.intake_condition.value,
        form.intake_type.value,
        form.breed_1.value,
        form.colour.value,
    ];
    if (currentAnimal === 'dog') required.push(form.size.value);
    if (required.some(v => !v)) {
        const msg = document.getElementById('error-msg');
        msg.textContent = currentAnimal === 'dog'
            ? 'Please fill in all fields — city, age, sex, spay/neuter, intake date, condition, type, breed, size, and colour.'
            : 'Please fill in all fields — city, age, sex, spay/neuter, intake date, condition, type, breed, and colour.';
        msg.style.display = 'block';
        return;
    }

    const data = {
        animal_species:   form.animal_species.value,
        city:             form.city.value.trim(),
        age_intake:       parseFloat(form.age_intake.value),
        sex:              form.sex.value,
        spay_neuter:      form.spay_neuter.value,
        intake_month:     parseInt(form.intake_month.value),
        intake_day:       parseInt(form.intake_day.value),
        intake_year:      parseInt(form.intake_year.value),
        intake_condition: form.intake_condition.value,
        intake_type:      form.intake_type.value,
        breed_1:          form.breed_1.value || null,
        is_mixed:         form.is_mixed.checked,
        breed_2:          form.is_mixed.checked ? (form.breed_2.value || 'None') : 'None',
        size:             currentAnimal === 'dog' ? (form.size.value || null) : null,
        colour:           form.colour.value || null,
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const body = await response.json().catch(() => ({}));
            const msg = document.getElementById('error-msg');
            if (response.status === 503 || body.error === 'high_demand') {
                msg.textContent = '🐾 Our server is experiencing high demand right now — please try again in a few minutes!';
            } else {
                console.error('Server error:', response.status, body);
                msg.textContent = `Server error ${response.status} — check the terminal for details.`;
            }
            msg.style.display = 'block';
            return;
        }

        const result = await response.json();
        console.log('Prediction result:', result);

        document.getElementById('result-days').textContent = result.predicted_bin;
        document.getElementById('result-emoji').textContent = currentAnimal === 'cat' ? '🐱' : '🐶';
        showPage('page-result');
    } catch (err) {
        console.error('Fetch failed:', err);
        const msg = document.getElementById('error-msg');
        msg.textContent = 'Could not reach the server — make sure app.py is running.';
        msg.style.display = 'block';
    }
});
