/**
 * Create one unpublished FairVision doctor-evaluation form.
 *
 * Before running:
 * 1. Upload form_cases.csv and all case_XXX.jpg images to one Drive folder.
 * 2. Paste that folder's ID into DRIVE_FOLDER_ID below.
 * 3. Optionally add the professor's email as an editor.
 * 4. Run createFairVisionPilotForm from script.google.com.
 *
 * The CSV must be the blinded output of prepare_google_form_pilot.py.
 */

const DRIVE_FOLDER_ID = "PASTE_GOOGLE_DRIVE_FOLDER_ID_HERE";
const PROFESSOR_EMAIL = ""; // Optional: professor@university.edu
const FORM_TITLE = "FairVision Human Evaluation – Pilot";
const RESPONSE_SHEET_TITLE = "FairVision Human Evaluation – Pilot Responses";


function createFairVisionPilotForm() {
  if (DRIVE_FOLDER_ID === "PASTE_GOOGLE_DRIVE_FOLDER_ID_HERE") {
    throw new Error("Set DRIVE_FOLDER_ID before running the script.");
  }
  const folder = DriveApp.getFolderById(DRIVE_FOLDER_ID);
  const rows = readCsvFromFolder_(folder, "form_cases.csv");
  validateRows_(rows);

  // Start unpublished so the professor can review before doctors receive it.
  const form = FormApp.create(FORM_TITLE, false);
  form.setDescription(
    "Purpose: independently evaluate a binary ophthalmic diagnosis from " +
    "de-identified imaging and the displayed demographic context.\n\n" +
    "Review each case using only the supplied information. Select a binary " +
    "diagnosis, rate confidence, and rate whether image quality is adequate."
  );
  form.setProgressBar(true);
  form.setConfirmationMessage("Thank you. Your evaluation has been recorded.");
  form.setCollectEmail(false);

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
    .setChoiceValues(["< 2", "2–5", "6–10", "11–20", "> 20"])
    .setRequired(true);

  rows.forEach(function(row, index) {
    const diseaseName = diseaseDisplayName_(row.disease);
    const demographics = [
      "Age: " + row.age,
      "Gender: " + row.gender,
      "Race: " + row.race,
      "Ethnicity: " + row.ethnicity
    ].join("\n");

    form.addPageBreakItem()
      .setTitle("Case " + padNumber_(index + 1, 2) + " — " + row.case_id)
      .setHelpText(demographics);

    const imageFile = getUniqueFile_(folder, row.image_filename);
    form.addImageItem()
      .setTitle("Ophthalmic imaging")
      .setHelpText("SLO/fundus and representative OCT B-scans")
      .setImage(imageFile.getBlob());

    form.addMultipleChoiceItem()
      .setTitle("What is your diagnosis for this case?")
      .setChoiceValues(["0 — No " + diseaseName, "1 — " + diseaseName])
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

  const responseSheet = SpreadsheetApp.create(RESPONSE_SHEET_TITLE);
  form.setDestination(FormApp.DestinationType.SPREADSHEET, responseSheet.getId());
  if (PROFESSOR_EMAIL.trim()) {
    form.addEditor(PROFESSOR_EMAIL.trim());
    DriveApp.getFileById(responseSheet.getId()).addEditor(PROFESSOR_EMAIL.trim());
  }

  // Keep it unpublished until the professor approves it. This guard supports
  // Forms runtimes where setPublished is present.
  if (typeof form.setPublished === "function") {
    form.setPublished(false);
  }

  console.log("Form edit URL: " + form.getEditUrl());
  console.log("Form respondent URL (unavailable until published): " + form.getPublishedUrl());
  console.log("Response sheet URL: " + responseSheet.getUrl());
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


function validateRows_(rows) {
  const required = [
    "review_order", "case_id", "image_filename", "age", "gender",
    "race", "ethnicity", "disease"
  ];
  if (rows.length !== 50) {
    throw new Error("Expected exactly 50 cases, found " + rows.length + ".");
  }
  const seen = {};
  rows.forEach(function(row, index) {
    required.forEach(function(column) {
      if (!(column in row) || String(row[column]).trim() === "") {
        throw new Error("Row " + (index + 2) + " is missing " + column + ".");
      }
    });
    if (seen[row.case_id]) {
      throw new Error("Duplicate case_id: " + row.case_id);
    }
    seen[row.case_id] = true;
  });
}


function getUniqueFile_(folder, filename) {
  const files = folder.getFilesByName(filename);
  if (!files.hasNext()) {
    throw new Error("Missing Drive file: " + filename);
  }
  const file = files.next();
  if (files.hasNext()) {
    throw new Error("More than one Drive file is named: " + filename);
  }
  return file;
}


function diseaseDisplayName_(value) {
  const normalized = String(value).trim().toLowerCase();
  if (normalized === "amd") return "AMD";
  if (normalized === "dr") return "DR";
  if (normalized === "glaucoma") return "glaucoma";
  throw new Error("Unsupported disease: " + value);
}


function padNumber_(value, width) {
  return String(value).padStart(width, "0");
}
