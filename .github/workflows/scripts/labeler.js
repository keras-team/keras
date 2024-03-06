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
 *  context Information about the workflow run.
 */

module.exports = async ({ github, context }) => {
    const issue_title = context.payload.issue ?  context.payload.issue.title : context.payload.pull_request.title
    const issue_discription = context.payload.issue ? context.payload.issue.body : context.payload.pull_request.body
    const issue_number = context.payload.issue ? context.payload.issue.number : context.payload.pull_request.number
    const keyword_label =  {
         gemma:'Gemma'
    }
    const labelsToAdd = []
    console.log(issue_title,issue_discription,issue_number)
    
    for(const [keyword, label] of Object.entries(keyword_label)){
     if(issue_title.toLowerCase().indexOf(keyword) !=-1 || issue_discription.toLowerCase().indexOf(keyword) !=-1 ){
        console.log(`'${keyword}'keyword is present inside the title or description. Pushing label '${label}' to row.`)
        labelsToAdd.push(label)
    }
   }
   if(labelsToAdd.length > 0){
    console.log(`Adding labels ${labelsToAdd} to the issue '#${issue_number}'.`)
     github.rest.issues.addLabels({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        labels: labelsToAdd
     })
   }
};