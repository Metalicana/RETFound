/**
 * Create four FairVision doctor-evaluation Google Forms.
 *
 * Before running:
 * 1. Prepare four bundles with prepare_google_form_pilot.py --disease mixed.
 * 2. Upload each bundle's form_cases.csv and 15 JPG files to its own Drive folder.
 * 3. Paste the four Drive folder IDs into FORM_CONFIGS below.
 * 4. Run createFairVisionForm1 through createFairVisionForm4 separately.
 *
 * Every form is ordered glaucoma -> AMD -> DR and contains five cases from
 * each disease. Source filenames and case IDs are never displayed.
 */

const PROFESSOR_EMAIL = ""; // Optional: professor@university.edu

const FORM_CONFIGS = [
  {
    folderId: "PASTE_MANIFEST_01_DRIVE_FOLDER_ID",
    formTitle: "FairVision Human Evaluation - Form 1",
    responseSheetTitle: "FairVision Human Evaluation - Form 1 Responses"
  },
  {
    folderId: "PASTE_MANIFEST_02_DRIVE_FOLDER_ID",
    formTitle: "FairVision Human Evaluation - Form 2",
    responseSheetTitle: "FairVision Human Evaluation - Form 2 Responses"
  },
  {
    folderId: "PASTE_MANIFEST_03_DRIVE_FOLDER_ID",
    formTitle: "FairVision Human Evaluation - Form 3",
    responseSheetTitle: "FairVision Human Evaluation - Form 3 Responses"
  },
  {
    folderId: "PASTE_MANIFEST_04_DRIVE_FOLDER_ID",
    formTitle: "FairVision Human Evaluation - Form 4",
    responseSheetTitle: "FairVision Human Evaluation - Form 4 Responses"
  }
];


function createFairVisionForm1() {
  createFairVisionMixedForm_(0);
}


function createFairVisionForm2() {
  createFairVisionMixedForm_(1);
}


function createFairVisionForm3() {
  createFairVisionMixedForm_(2);
}


function createFairVisionForm4() {
  createFairVisionMixedForm_(3);
}


// This convenience function may exceed Apps Script's execution-time limit if
// the uploaded images are large. Running the four functions above separately
// is safer and makes failures easier to retry.
function createAllFairVisionForms() {
  for (let index = 0; index < FORM_CONFIGS.length; index++) {
    createFairVisionMixedForm_(index);
  }
}


function createFairVisionMixedForm_(configIndex) {
  const config = FORM_CONFIGS[configIndex];
  if (!config) {
    throw new Error("Invalid form configuration index: " + configIndex);
  }
  if (!config.folderId || config.folderId.indexOf("PASTE_") === 0) {
    throw new Error("Set the Drive folder ID for Form " + (configIndex + 1) + ".");
  }

  const folder = DriveApp.getFolderById(config.folderId);
  const rows = readCsvFromFolder_(folder, "form_cases.csv");
  validateAndOrderRows_(rows);

  // Forms start unpublished so they can be reviewed before distribution.
  const form = FormApp.create(config.formTitle, false);
  form.setDescription(
    "Purpose: independently evaluate ophthalmic diagnoses from de-identified " +
    "imaging and demographic context.\n\n" +
    "This form contains five glaucoma cases, followed by five age-related " +
    "macular degeneration cases, followed by five diabetic retinopathy cases. " +
    "For every case, select a binary diagnosis, rate your confidence, and " +
    "indicate whether the supplied imaging is adequate."
  );
  form.setProgressBar(true);
  form.setConfirmationMessage("Thank you. Your evaluation has been recorded.");
  form.setCollectEmail(false);

  addReviewerQuestions_(form);

  rows.forEach(function(row, index) {
    const diseaseName = diseaseDisplayName_(row.disease);
    const caseNumber = padNumber_(index + 1, 2);
    const caseWithinDisease = (index % 5) + 1;
    const demographics = [
      "Imaging modality: " + row.imaging_modality,
      "Age: " + row.age,
      "Gender: " + row.gender,
      "Race: " + row.race,
      "Ethnicity: " + row.ethnicity
    ].join("\n");

    form.addPageBreakItem()
      .setTitle(
        "Case " + caseNumber + " - " + diseaseName + " " + caseWithinDisease + " of 5"
      )
      .setHelpText(demographics);

    const imageFile = getUniqueFile_(folder, row.image_filename);
    form.addImageItem()
      .setTitle(row.imaging_modality)
      .setHelpText("Review all displayed imaging before answering.")
      .setImage(imageFile.getBlob());

    const labels = diagnosisLabels_(row.disease);
    form.addMultipleChoiceItem()
      .setTitle("What is your diagnosis for this case?")
      .setChoiceValues(["0 - " + labels.negative, "1 - " + labels.positive])
      .setRequired(true);

    form.addScaleItem()
      .setTitle("How confident are you in this diagnosis?")
      .setBounds(1, 5)
      .setLabels("Very uncertain", "Very confident")
      .setRequired(true);

    form.addMultipleChoiceItem()
      .setTitle("Is the supplied imaging adequate for diagnosis?")
      .setChoiceValues(["Adequate", "Partially adequate", "Inadequate"])
      .setRequired(true);
  });

  const responseSheet = SpreadsheetApp.create(config.responseSheetTitle);
  form.setDestination(FormApp.DestinationType.SPREADSHEET, responseSheet.getId());
  if (PROFESSOR_EMAIL.trim()) {
    form.addEditor(PROFESSOR_EMAIL.trim());
    DriveApp.getFileById(responseSheet.getId()).addEditor(PROFESSOR_EMAIL.trim());
  }

  if (typeof form.setPublished === "function") {
    form.setPublished(false);
  }

  console.log("Created " + config.formTitle);
  console.log("Form edit URL: " + form.getEditUrl());
  console.log("Response sheet URL: " + responseSheet.getUrl());
}


function addReviewerQuestions_(form) {
  form.addTextItem()
    .setTitle("Reviewer ID")
    .setHelpText("Use the pseudonymous reviewer ID assigned by the study team.")
    .setRequired(true);
  form.addListItem()
    .setTitle("Clinical specialty")
    .setChoiceValues([
      "Comprehensive ophthalmology",
      "Glaucoma",
      "Retina",
      "Other ophthalmology",
      "Optometry",
      "Other"
    ])
    .setRequired(true);
  form.addListItem()
    .setTitle("Years of ophthalmic clinical experience")
    .setChoiceValues(["< 2", "2-5", "6-10", "11-20", "> 20"])
    .setRequired(true);
}


function readCsvFromFolder_(folder, filename) {
  const file = getUniqueFile_(folder, filename);
  const values = Utilities.parseCsv(file.getBlob().getDataAsString("UTF-8"));
  if (values.length < 2) {
    throw new Error(filename + " has no case rows.");
  }
  const headers = values[0].map(function(value) { return value.trim(); });
  return values.slice(1).filter(function(row) {
    return row.some(function(value) { return String(value).trim() !== ""; });
  }).map(function(valuesRow) {
    const row = {};
    headers.forEach(function(header, index) { row[header] = valuesRow[index] || ""; });
    return row;
  });
}


function validateAndOrderRows_(rows) {
  const required = [
    "review_order", "image_filename", "age", "gender", "race",
    "ethnicity", "disease", "imaging_modality"
  ];
  if (rows.length !== 15) {
    throw new Error("Expected exactly 15 cases, found " + rows.length + ".");
  }

  const diseaseOrder = {"glaucoma": 0, "amd": 1, "dr": 2};
  const counts = {"glaucoma": 0, "amd": 0, "dr": 0};
  const seenImages = {};
  rows.forEach(function(row, index) {
    required.forEach(function(column) {
      if (!(column in row) || String(row[column]).trim() === "") {
        throw new Error("CSV row " + (index + 2) + " is missing " + column + ".");
      }
    });
    row.disease = String(row.disease).trim().toLowerCase();
    if (!(row.disease in diseaseOrder)) {
      throw new Error("Unsupported disease in row " + (index + 2) + ": " + row.disease);
    }
    counts[row.disease]++;
    if (seenImages[row.image_filename]) {
      throw new Error("Duplicate image_filename: " + row.image_filename);
    }
    seenImages[row.image_filename] = true;
  });
  ["glaucoma", "amd", "dr"].forEach(function(disease) {
    if (counts[disease] !== 5) {
      throw new Error("Expected 5 " + disease + " cases, found " + counts[disease] + ".");
    }
  });

  rows.sort(function(left, right) {
    const diseaseDifference = diseaseOrder[left.disease] - diseaseOrder[right.disease];
    if (diseaseDifference !== 0) return diseaseDifference;
    return Number(left.review_order) - Number(right.review_order);
  });
}


function getUniqueFile_(folder, filename) {
  const files = folder.getFilesByName(filename);
  if (!files.hasNext()) {
    throw new Error("Missing Drive image required by the form bundle.");
  }
  const file = files.next();
  if (files.hasNext()) {
    throw new Error("More than one uploaded image has the same bundle name.");
  }
  return file;
}


function diseaseDisplayName_(value) {
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "glaucoma") return "Glaucoma";
  if (normalized === "amd") return "Age-related macular degeneration (AMD)";
  if (normalized === "dr") return "Diabetic retinopathy (DR)";
  throw new Error("Unsupported disease: " + value);
}


function diagnosisLabels_(value) {
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "glaucoma") {
    return {negative: "No glaucoma", positive: "Glaucoma"};
  }
  if (normalized === "amd") {
    return {negative: "No AMD", positive: "AMD"};
  }
  if (normalized === "dr") {
    return {negative: "No diabetic retinopathy", positive: "Diabetic retinopathy"};
  }
  throw new Error("Unsupported disease: " + value);
}


function padNumber_(value, width) {
  return String(value).padStart(width, "0");
}
