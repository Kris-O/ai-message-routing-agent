from enum import Enum
from typing import Annotated

from pydantic import BaseModel, EmailStr, StringConstraints


class TargetDepartment(str, Enum):
    KADRY = "KADRY"
    HR = "HR"
    HELPDESK = "HELPDESK"
    IT = "IT"
    INNE = "INNE"


# Target addresses exactly as given in the task brief (the five-address list). The internal Enum labels
# (KADRY/HR/HELPDESK/IT/INNE) map onto them; INNE is the brief's `other@` fallback.
DEPARTMENT_EMAILS: dict[TargetDepartment, str] = {
    TargetDepartment.KADRY: "kadry@example.com",
    TargetDepartment.HR: "human-resources@example.com",
    TargetDepartment.HELPDESK: "help-desk@example.com",
    TargetDepartment.IT: "it@example.com",
    TargetDepartment.INNE: "other@example.com",
}


class RouteRequest(BaseModel):
    email: EmailStr
    message: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    ]


class RouteResponse(BaseModel):
    department: TargetDepartment
    target_email: str
    sent: bool
    detail: str
