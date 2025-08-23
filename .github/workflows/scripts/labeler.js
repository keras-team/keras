/*
Copyright 2024 Google LLC. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/**
 * Invoked from labeler.yaml file to add
 * label 'Gemma' to the issue and PR for which have gemma keyword present.
 * @param {!Object.<string,!Object>} github contains pre defined functions.
 * context Information about the workflow run.
 */

module.exports = async ({ github, context }) => {

    // Determine if the event is an issue or a pull request.
    const isIssue = !!context.payload.issue;

    // Get the issue/PR title, description, and number from the payload.
    // Use an empty string for the description if it's null to prevent runtime errors.
    const issue_title = isIssue ? context.payload.issue.title : context.payload.pull_request.title;
    const issue_description = (isIssue ? context.payload.issue.body : context.payload.pull_request.body) || '';
    const issue_number = isIssue ? context.payload.issue.number : context.payload.pull_request.number;
    
    // Define the keyword-to-label mapping.
    const keyword_label =  {
         gemma: 'Gemma'
    };
    // Array to hold labels that need to be added.
    const labelsToAdd = [];

    console.log(`Processing event for issue/PR #${issue_number}: "${issue_title}"`);
    
    // Loop through the keywords and check if they exist in the title or description.
    for (const [keyword, label] of Object.entries(keyword_label)) {
        // Use .includes() for a cleaner and more modern check.
        if (issue_title.toLowerCase().includes(keyword) || issue_description.toLowerCase().includes(keyword)) {
            console.log(`'${keyword}' keyword is present in the title or description. Pushing label '${label}' to the array.`);
            labelsToAdd.push(label);
        }
    }

    // Add labels if the labelsToAdd array is not empty.
    if (labelsToAdd.length > 0) {
        console.log(`Adding labels ${labelsToAdd} to issue/PR '#${issue_number}'.`);
        
        try {
            // Await the asynchronous API call to ensure it completes.
            await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: issue_number, // Use the correct issue_number variable
                labels: labelsToAdd
            });
            console.log(`Successfully added labels.`);
        } catch (error) {
            console.error(`Failed to add labels: ${error.message}`);
        }
    } else {
        console.log("No matching keywords found. No labels to add.");
    }
};
