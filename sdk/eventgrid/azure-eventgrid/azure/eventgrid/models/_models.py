# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) Python Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------
# pylint: disable=useless-super-delegation

import datetime
from typing import Any, List, Mapping, Optional, TYPE_CHECKING, overload

from azure.core.exceptions import ODataV4Format

from .. import _model_base
from .._model_base import rest_field

if TYPE_CHECKING:
    from .. import models as _models


class AcknowledgeResult(_model_base.Model):
    """The result of the Acknowledge operation.


    :ivar failed_lock_tokens: Array of FailedLockToken for failed cloud events. Each
     FailedLockToken includes the lock token along with the related error information (namely, the
     error code and description). Required.
    :vartype failed_lock_tokens: list[~azure.eventgrid.models.FailedLockToken]
    :ivar succeeded_lock_tokens: Array of lock tokens for the successfully acknowledged cloud
     events. Required.
    :vartype succeeded_lock_tokens: list[str]
    """

    failed_lock_tokens: List["_models.FailedLockToken"] = rest_field(name="failedLockTokens")
    """Array of FailedLockToken for failed cloud events. Each FailedLockToken includes the lock token
     along with the related error information (namely, the error code and description). Required."""
    succeeded_lock_tokens: List[str] = rest_field(name="succeededLockTokens")
    """Array of lock tokens for the successfully acknowledged cloud events. Required."""

    @overload
    def __init__(
        self,
        *,
        failed_lock_tokens: List["_models.FailedLockToken"],
        succeeded_lock_tokens: List[str],
    ) -> None: ...

    @overload
    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """
        :param mapping: raw JSON to initialize the model.
        :type mapping: Mapping[str, Any]
        """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class BrokerProperties(_model_base.Model):
    """Properties of the Event Broker operation.


    :ivar lock_token: The token of the lock on the event. Required.
    :vartype lock_token: str
    :ivar delivery_count: The attempt count for delivering the event. Required.
    :vartype delivery_count: int
    """

    lock_token: str = rest_field(name="lockToken")
    """The token of the lock on the event. Required."""
    delivery_count: int = rest_field(name="deliveryCount")
    """The attempt count for delivering the event. Required."""


class CloudEvent(_model_base.Model):
    """Properties of an event published to an Azure Messaging EventGrid Namespace topic using the
    CloudEvent 1.0 Schema.


    :ivar id: An identifier for the event. The combination of id and source must be unique for each
     distinct event. Required.
    :vartype id: str
    :ivar source: Identifies the context in which an event happened. The combination of id and
     source must be unique for each distinct event. Required.
    :vartype source: str
    :ivar data: Event data specific to the event type.
    :vartype data: any
    :ivar data_base64: Event data specific to the event type, encoded as a base64 string.
    :vartype data_base64: bytes
    :ivar type: Type of event related to the originating occurrence. Required.
    :vartype type: str
    :ivar time: The time (in UTC) the event was generated, in RFC3339 format.
    :vartype time: ~datetime.datetime
    :ivar specversion: The version of the CloudEvents specification which the event uses. Required.
    :vartype specversion: str
    :ivar dataschema: Identifies the schema that data adheres to.
    :vartype dataschema: str
    :ivar datacontenttype: Content type of data value.
    :vartype datacontenttype: str
    :ivar subject: This describes the subject of the event in the context of the event producer
     (identified by source).
    :vartype subject: str
    """

    id: str = rest_field()
    """An identifier for the event. The combination of id and source must be unique for each distinct
     event. Required."""
    source: str = rest_field()
    """Identifies the context in which an event happened. The combination of id and source must be
     unique for each distinct event. Required."""
    data: Optional[Any] = rest_field()
    """Event data specific to the event type."""
    data_base64: Optional[bytes] = rest_field(format="base64")
    """Event data specific to the event type, encoded as a base64 string."""
    type: str = rest_field()
    """Type of event related to the originating occurrence. Required."""
    time: Optional[datetime.datetime] = rest_field(format="rfc3339")
    """The time (in UTC) the event was generated, in RFC3339 format."""
    specversion: str = rest_field()
    """The version of the CloudEvents specification which the event uses. Required."""
    dataschema: Optional[str] = rest_field()
    """Identifies the schema that data adheres to."""
    datacontenttype: Optional[str] = rest_field()
    """Content type of data value."""
    subject: Optional[str] = rest_field()
    """This describes the subject of the event in the context of the event producer (identified by
     source)."""


class FailedLockToken(_model_base.Model):
    """Failed LockToken information.


    :ivar lock_token: The lock token of an entry in the request. Required.
    :vartype lock_token: str
    :ivar error: Error information of the failed operation result for the lock token in the
     request. Required.
    :vartype error: ~azure.core.ODataV4Format
    """

    lock_token: str = rest_field(name="lockToken")
    """The lock token of an entry in the request. Required."""
    error: ODataV4Format = rest_field()
    """Error information of the failed operation result for the lock token in the request. Required."""

    @overload
    def __init__(
        self,
        *,
        lock_token: str,
        error: ODataV4Format,
    ) -> None: ...

    @overload
    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """
        :param mapping: raw JSON to initialize the model.
        :type mapping: Mapping[str, Any]
        """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class PublishResult(_model_base.Model):
    """The result of the Publish operation."""


class ReceiveDetails(_model_base.Model):
    """Receive operation details per Cloud Event.


    :ivar broker_properties: The Event Broker details. Required.
    :vartype broker_properties: ~azure.eventgrid.models._models.BrokerProperties
    :ivar event: Cloud Event details. Required.
    :vartype event: ~azure.eventgrid.models._models.CloudEvent
    """

    broker_properties: "_models._models.BrokerProperties" = rest_field(name="brokerProperties")
    """The Event Broker details. Required."""
    event: "_models._models.CloudEvent" = rest_field()
    """Cloud Event details. Required."""


class ReceiveResult(_model_base.Model):
    """Details of the Receive operation response.


    :ivar details: Array of receive responses, one per cloud event. Required.
    :vartype details: list[~azure.eventgrid.models._models.ReceiveDetails]
    """

    details: List["_models._models.ReceiveDetails"] = rest_field(name="value")
    """Array of receive responses, one per cloud event. Required."""


class RejectResult(_model_base.Model):
    """The result of the Reject operation.


    :ivar failed_lock_tokens: Array of FailedLockToken for failed cloud events. Each
     FailedLockToken includes the lock token along with the related error information (namely, the
     error code and description). Required.
    :vartype failed_lock_tokens: list[~azure.eventgrid.models.FailedLockToken]
    :ivar succeeded_lock_tokens: Array of lock tokens for the successfully rejected cloud events.
     Required.
    :vartype succeeded_lock_tokens: list[str]
    """

    failed_lock_tokens: List["_models.FailedLockToken"] = rest_field(name="failedLockTokens")
    """Array of FailedLockToken for failed cloud events. Each FailedLockToken includes the lock token
     along with the related error information (namely, the error code and description). Required."""
    succeeded_lock_tokens: List[str] = rest_field(name="succeededLockTokens")
    """Array of lock tokens for the successfully rejected cloud events. Required."""

    @overload
    def __init__(
        self,
        *,
        failed_lock_tokens: List["_models.FailedLockToken"],
        succeeded_lock_tokens: List[str],
    ) -> None: ...

    @overload
    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """
        :param mapping: raw JSON to initialize the model.
        :type mapping: Mapping[str, Any]
        """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class ReleaseResult(_model_base.Model):
    """The result of the Release operation.


    :ivar failed_lock_tokens: Array of FailedLockToken for failed cloud events. Each
     FailedLockToken includes the lock token along with the related error information (namely, the
     error code and description). Required.
    :vartype failed_lock_tokens: list[~azure.eventgrid.models.FailedLockToken]
    :ivar succeeded_lock_tokens: Array of lock tokens for the successfully released cloud events.
     Required.
    :vartype succeeded_lock_tokens: list[str]
    """

    failed_lock_tokens: List["_models.FailedLockToken"] = rest_field(name="failedLockTokens")
    """Array of FailedLockToken for failed cloud events. Each FailedLockToken includes the lock token
     along with the related error information (namely, the error code and description). Required."""
    succeeded_lock_tokens: List[str] = rest_field(name="succeededLockTokens")
    """Array of lock tokens for the successfully released cloud events. Required."""

    @overload
    def __init__(
        self,
        *,
        failed_lock_tokens: List["_models.FailedLockToken"],
        succeeded_lock_tokens: List[str],
    ) -> None: ...

    @overload
    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """
        :param mapping: raw JSON to initialize the model.
        :type mapping: Mapping[str, Any]
        """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class RenewLocksResult(_model_base.Model):
    """The result of the RenewLock operation.


    :ivar failed_lock_tokens: Array of FailedLockToken for failed cloud events. Each
     FailedLockToken includes the lock token along with the related error information (namely, the
     error code and description). Required.
    :vartype failed_lock_tokens: list[~azure.eventgrid.models.FailedLockToken]
    :ivar succeeded_lock_tokens: Array of lock tokens for the successfully renewed locks. Required.
    :vartype succeeded_lock_tokens: list[str]
    """

    failed_lock_tokens: List["_models.FailedLockToken"] = rest_field(name="failedLockTokens")
    """Array of FailedLockToken for failed cloud events. Each FailedLockToken includes the lock token
     along with the related error information (namely, the error code and description). Required."""
    succeeded_lock_tokens: List[str] = rest_field(name="succeededLockTokens")
    """Array of lock tokens for the successfully renewed locks. Required."""

    @overload
    def __init__(
        self,
        *,
        failed_lock_tokens: List["_models.FailedLockToken"],
        succeeded_lock_tokens: List[str],
    ) -> None: ...

    @overload
    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """
        :param mapping: raw JSON to initialize the model.
        :type mapping: Mapping[str, Any]
        """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
