from enum import Enum
from typing import Annotated

from pydantic import BaseModel, EmailStr, StringConstraints


class TargetDepartment(str, Enum):
    KADRY = "KADRY"
    HR = "HR"
    HELPDESK = "HELPDESK"
    IT = "IT"
    INNE = "INNE"


DEPARTMENT_EMAILS: dict[TargetDepartment, str] = {
    TargetDepartment.KADRY: "kadry@firma.pl",
    TargetDepartment.HR: "hr@firma.pl",
    TargetDepartment.HELPDESK: "helpdesk@firma.pl",
    TargetDepartment.IT: "it@firma.pl",
    TargetDepartment.INNE: "kontakt@firma.pl",
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
