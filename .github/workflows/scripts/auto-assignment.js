/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/** Automatically assign issues and PRs to users in the `assigneesList` 
 *  on a rotating basis.

  @param {!object}
    GitHub objects can call GitHub APIs using their built-in library functions.
    The context object contains issue and PR details.
*/

module.exports = async ({ github, context }) => {
  let issueNumber;
  let assigneesList;
  // Is this an issue? If so, assign the issue number. Otherwise, assign the PR number.
  if (context.payload.issue) {
    //assignee List for issues. 
    assigneesList = ["mehtamansi29", "sachinprasadhs"];
    issueNumber = context.payload.issue.number;
  } else {
    //assignee List for PRs. 
    assigneesList = [];
    issueNumber = context.payload.number;
  }
  console.log("assignee list", assigneesList);
  console.log("entered auto assignment for this issue:  ", issueNumber);
  if (!assigneesList.length) {
    console.log("No assignees found for this repo.");
    return;
  }
  let noOfAssignees = assigneesList.length;
  let selection = issueNumber % noOfAssignees;
  let assigneeForIssue = assigneesList[selection];

  console.log(
    "issue Number = ",
    issueNumber + " , assigning to: ",
    assigneeForIssue
  );
  return github.rest.issues.addAssignees({
    issue_number: context.issue.number,
    owner: context.repo.owner,
    repo: context.repo.repo,
    assignees: [assigneeForIssue],
  });
};
